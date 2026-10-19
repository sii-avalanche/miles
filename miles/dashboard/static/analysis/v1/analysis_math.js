/* Render the exported prose with the same KaTeX engine used by AvaVis. */
(() => {
  if (typeof window.renderMathInElement !== "function") return;
  const diagnostics = window.__avaAnalysisMath = { rendered: 0, errors: 0, queued: 0 };
  const options = {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
      { left: "$", right: "$", display: false },
    ],
    ignoredTags: ["script", "noscript", "style", "textarea", "code", "option", "math"],
    throwOnError: false,
    strict: "ignore",
    trust: false,
    errorCallback: () => { diagnostics.errors++; },
  };
  const queue = [];
  let scheduled = false;
  const idle = window.requestIdleCallback
    ? callback => window.requestIdleCallback(callback, { timeout: 150 })
    : callback => setTimeout(() => callback({ timeRemaining: () => 0 }), 0);

  function protectCode(element) {
    if (element.children.length) return;
    const parts = element.textContent.split(/(\x60{3}[\s\S]*?\x60{3}|\x60[^\x60\n]+\x60)/g);
    if (parts.length === 1) return;
    const fragment = document.createDocumentFragment();
    parts.forEach((part, index) => {
      if (index % 2) {
        const code = document.createElement("code");
        code.className = "math-code";
        code.textContent = part;
        fragment.append(code);
      } else {
        fragment.append(document.createTextNode(part));
      }
    });
    element.replaceChildren(fragment);
  }

  function chunks(text) {
    // Keep complete formulas and fenced code together while splitting long prose.
    const protectedParts = /(\x60{3}[\s\S]*?\x60{3}|\x60[^\x60\n]+\x60|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$|(?<![\\$])\$(?!\$)[^\n$]*?\$(?!\$))/g;
    const result = [];
    let pending = "";
    let offset = 0;
    function append(value, protectedPart = false) {
      for (const part of protectedPart ? [value] : value.split(/(\n)/)) {
        pending += part;
        if (pending.length >= 8000) {
          result.push(pending);
          pending = "";
        }
      }
    }
    for (const match of text.matchAll(protectedParts)) {
      append(text.slice(offset, match.index));
      append(match[0], true);
      offset = match.index + match[0].length;
    }
    append(text.slice(offset));
    if (pending) result.push(pending);
    return result;
  }

  function render(element) {
    protectCode(element);
    window.renderMathInElement(element, options);
    element.dataset.mathRendered = "1";
    diagnostics.rendered++;
  }

  function pump(deadline) {
    scheduled = false;
    let count = 0;
    do {
      render(queue.shift());
      count++;
    } while (queue.length && count < 4 && deadline.timeRemaining() > 8);
    diagnostics.queued = queue.length;
    if (queue.length) schedule();
  }

  function schedule() {
    if (!scheduled && queue.length) {
      scheduled = true;
      idle(pump);
    }
  }

  const observer = "IntersectionObserver" in window ? new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        observer.unobserve(entry.target);
        queue.push(entry.target);
      }
    }
    diagnostics.queued = queue.length;
    schedule();
  }, { rootMargin: "300px" }) : null;

  for (const element of document.querySelectorAll(".prompt, pre, td")) {
    if (element.closest("details, code, .katex") || !/[\\$]/.test(element.textContent)) continue;
    if (element.parentElement.closest(".prompt, pre, td")) continue;
    element.classList.add("math-text");
    let targets = [element];
    if (!element.children.length && element.textContent.length > 24000) {
      const fragment = document.createDocumentFragment();
      targets = chunks(element.textContent).map(part => {
        const span = document.createElement("span");
        span.textContent = part;
        fragment.append(span);
        return span;
      });
      element.replaceChildren(fragment);
    }
    for (const target of targets) {
      if (observer) observer.observe(target);
      else queue.push(target);
    }
  }
  if (!observer) schedule();
})();
