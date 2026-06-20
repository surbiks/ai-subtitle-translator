"use strict";

const form = document.getElementById("translate-form");
const submitBtn = document.getElementById("submit-btn");

const progressEl = document.getElementById("progress");
const barFill = document.getElementById("bar-fill");
const progressText = document.getElementById("progress-text");
const failedNote = document.getElementById("failed-note");

const resultEl = document.getElementById("result");
const downloadLink = document.getElementById("download-link");
const errorEl = document.getElementById("error");

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function resetUi() {
  hide(resultEl);
  hide(errorEl);
  hide(failedNote);
  show(progressEl);
  barFill.style.width = "0%";
  progressText.textContent = "Starting…";
}

function setProgress(done, total, failed) {
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  barFill.style.width = pct + "%";
  progressText.textContent = `Translating… ${done}/${total} chunks (${pct}%)`;
  if (failed > 0) {
    failedNote.textContent = `${failed} chunk(s) failed and kept the original text.`;
    show(failedNote);
  }
}

function finish(evt) {
  barFill.style.width = "100%";
  const translated = evt.translated ?? 0;
  const total = evt.total ?? 0;
  progressText.textContent = `Done — ${translated}/${total} chunks translated.`;
  downloadLink.href = evt.download_url;
  downloadLink.setAttribute("download", evt.filename || "translation.srt");
  show(resultEl);
  submitBtn.disabled = false;
}

function fail(message) {
  hide(progressEl);
  errorEl.textContent = "Error: " + message;
  show(errorEl);
  submitBtn.disabled = false;
}

function streamProgress(jobId) {
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  source.onmessage = (e) => {
    let evt;
    try { evt = JSON.parse(e.data); } catch { return; }
    if (evt.type === "progress") {
      setProgress(evt.done, evt.total, evt.failed);
    } else if (evt.type === "done") {
      finish(evt);
      source.close();
    } else if (evt.type === "error") {
      fail(evt.message);
      source.close();
    }
  };
  source.onerror = () => {
    // The stream closes normally after a terminal event; only surface an error
    // if we never reached a result.
    if (resultEl.classList.contains("hidden") && errorEl.classList.contains("hidden")) {
      fail("connection to the server was lost");
    }
    source.close();
  };
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  submitBtn.disabled = true;
  resetUi();

  try {
    const res = await fetch("/api/translate", {
      method: "POST",
      body: new FormData(form),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `request failed (${res.status})`);
    }
    const { job_id } = await res.json();
    streamProgress(job_id);
  } catch (err) {
    fail(err.message);
  }
});
