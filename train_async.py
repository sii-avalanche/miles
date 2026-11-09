import asyncio
import logging
import os

from miles.ray.placement_group import (
    create_rollout_components,
    create_training_models,
    maybe_start_api_server,
    update_weights,
)
from miles.ray.rollout.eval_dispatch import EvalDispatcher
from miles.utils.arguments import parse_args, validate_async_off_policy_correction
from miles.utils.async_utils import Disposer, eager_create_task, with_disposer
from miles.utils.data import remove_rollout_data_refs, remove_train_output_refs
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.misc import should_run_periodic_action
from miles.utils.orchestration_utils import init_orchestration_script

logger = logging.getLogger(__name__)


# The framework supports other asynchronous approaches such as fully async (see miles/rollout/fully_async_rollout.py).
async def train(args, *, disposer: Disposer, schedule_adapter=None):
    assert not args.colocate, "Colocation is not supported for async training."
    validate_async_off_policy_correction(args)
    _worker_manager = init_orchestration_script(args, disposer=disposer)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    inference_controller, rollout_executor, num_rollout_per_epoch = await create_rollout_components(args)
    disposer.add(inference_controller, rollout_executor)

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, rollout_executor)
    disposer.add(critic_model, actor_model)

    maybe_start_api_server(args, trainer_models={"actor": actor_model}, inference_controller=inference_controller)
    maybe_start_mini_ft_controller(args)

    # always update weight first so that sglang has the loaded weights from training.
    await _publish(args, actor_model, rollout_executor, inference_controller, schedule_adapter)

    if args.check_weight_update_equal:
        await inference_controller.check_weights(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )

    eval_dispatcher = EvalDispatcher(args, actor_model, rollout_executor)
    disposer.add(eval_dispatcher.drain)

    if args.eval_interval is not None and args.start_rollout_id == 0 and not args.skip_eval_before_train:
        await inference_controller.prepare_eval()
        await eval_dispatcher.dispatch(0, hf_dir=args.hf_checkpoint)

    async def save_training_model(model, rollout_id, force_sync):
        if args.use_critic and args.offload_train:
            await model.onload()
        await model.save_model(rollout_id, force_sync=force_sync)
        if args.use_critic and args.offload_train:
            await model.offload()

    async def prepare_and_generate(rollout_id):
        return await _prepare_and_generate(rollout_id, inference_controller, rollout_executor, schedule_adapter)

    await _train_loop(
        args,
        actor_model,
        critic_model,
        rollout_executor,
        inference_controller,
        eval_dispatcher,
        num_rollout_per_epoch,
        prepare_and_generate,
        save_training_model,
        schedule_adapter,
    )


async def _prepare_and_generate(rollout_id, inference, executor, adapter):
    if adapter is not None:
        adapter.rollout_started(rollout_id)
    await inference.prepare_rollout(rollout_id)
    pack = await executor.get(rollout_id)
    if adapter is not None:
        adapter.rollout_finished(rollout_id)
    return pack


async def _publish(args, actor, executor, inference, adapter, rollout_id=None):
    kwargs = {} if adapter is None else {"publication_context": adapter.publication_context()}
    if rollout_id is not None:
        kwargs["rollout_id"] = rollout_id
    result = await update_weights(args, actor, executor, inference, **kwargs)
    if adapter is not None:
        adapter.published(result)


async def _receive(args, future, adapter):
    pack = await future
    if adapter is not None:
        adapter.received(pack)
    return pack


async def _train_current(args, rollout_id, pack, actor, critic, adapter):
    if adapter is not None:
        adapter.before_train(rollout_id, pack)
    if args.use_critic:
        values = await critic.train(rollout_id, pack)
        if args.offload_train:
            await critic.offload()
        if rollout_id >= args.num_critic_only_steps:
            await actor.train(rollout_id, pack, external_data=values)
            if args.offload_train:
                await actor.offload()
        remove_train_output_refs(values)
    else:
        outputs = await actor.train(rollout_id, pack)
        if adapter is not None:
            try:
                adapter.after_train(rollout_id, pack, outputs)
            finally:
                remove_train_output_refs(outputs)
    remove_rollout_data_refs(args, pack)


async def _save_current(args, rollout_id, actor, critic, executor, save_model, external, adapter):
    context = adapter.before_save(rollout_id) if adapter is not None else None
    sentinel = os.stat(args.save_trigger_sentinel) if external and adapter is not None else None
    await save_model(actor, rollout_id, force_sync=external or rollout_id == args.num_rollout - 1)
    if args.use_critic:
        await save_model(critic, rollout_id, force_sync=external or rollout_id == args.num_rollout - 1)
    kwargs = {} if context is None else {"checkpoint_context": context}
    await executor.save(rollout_id, **kwargs)
    if adapter is not None:
        adapter.saved(rollout_id)
    if external and os.path.exists(args.save_trigger_sentinel):
        latest = os.stat(args.save_trigger_sentinel) if sentinel is not None else None
        if sentinel is None or (sentinel.st_ino, sentinel.st_mtime_ns) == (latest.st_ino, latest.st_mtime_ns):
            os.remove(args.save_trigger_sentinel)


async def _train_loop(args, actor, critic, executor, inference, evals, per_epoch, generate, save_model, adapter):
    next_future = None
    current = None
    try:
        if args.start_rollout_id < args.num_rollout:
            next_future = await eager_create_task(generate(args.start_rollout_id))
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            if next_future is not None:
                current = await _receive(args, next_future, adapter)
                next_future = None
            periodic = should_run_periodic_action(rollout_id, args.save_interval, per_epoch, args.num_rollout)
            external = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
            has_next = rollout_id + 1 < args.num_rollout
            update_due = (rollout_id + 1) % args.update_weights_interval == 0
            defer = args.fully_async and has_next and update_due
            stop_after = (
                args.debug_exit_after_rollout is not None
                and rollout_id - args.start_rollout_id + 1 >= args.debug_exit_after_rollout
            )
            if adapter is not None:
                adapter.request_save(periodic or external)
            prefetch = has_next and not defer
            if adapter is not None:
                prefetch = adapter.plan_next(prefetch and not stop_after)
            if prefetch:
                next_future = await eager_create_task(generate(rollout_id + 1))
            await _train_current(args, rollout_id, current, actor, critic, adapter)
            current = None
            external = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
            save_due = periodic or external
            if adapter is not None:
                adapter.request_save(save_due)
                save_due = adapter.save_requested and next_future is None
            if save_due:
                await _save_current(args, rollout_id, actor, critic, executor, save_model, external, adapter)
            if update_due:
                if not args.fully_async:
                    if next_future is not None:
                        current = await _receive(args, next_future, adapter)
                    next_future = None
                await _publish(args, actor, executor, inference, adapter, rollout_id)
                if defer:
                    next_future = await eager_create_task(generate(rollout_id + 1))
            if should_run_periodic_action(rollout_id, args.eval_interval, per_epoch, args.num_rollout):
                await inference.prepare_eval()
                await evals.dispatch(rollout_id, force=rollout_id == args.num_rollout - 1)
            if stop_after:
                logger.info("debug_exit_after_rollout reached at rollout_id=%d", rollout_id)
                break
            if has_next and current is None and next_future is None:
                # An empty checkpoint boundary restarts generation with the just-published actor.
                next_future = await eager_create_task(generate(rollout_id + 1))
    except BaseException as error:
        if adapter is not None:
            if current is not None and adapter.training is None:
                remove_rollout_data_refs(args, current)
            try:
                await adapter.failed(error, next_future, lambda pack: remove_rollout_data_refs(args, pack))
            except BaseException:
                logger.exception("Cleanup failed; preserving the originating training error")
        raise


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(with_disposer(train, args))
