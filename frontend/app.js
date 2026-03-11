/**
 * DatacortexAI — Frontend Application
 *
 * Client-side logic for file upload, WebSocket connection,
 * pipeline DAG visualization, and report rendering.
 */

(() => {
  'use strict';

  // ─── State ──────────────────────────────────────
  const state = {
    sessionId: null,
    file: null,
    ws: null,
    events: [],
    pipelineStartTime: null,
    reportMarkdown: '',
  };

  // ─── Elements ───────────────────────────────────
  const $ = (id) => document.getElementById(id);
  const uploadZone = $('upload-zone');
  const fileInput = $('file-input');
  const configPanel = $('config-panel');
  const fileName = $('file-name');
  const fileSize = $('file-size');
  const targetCol = $('target-col');
  const userQuery = $('user-query');
  const btnAnalyze = $('btn-analyze');
  const pipelineSection = $('pipeline-section');
  const pipelineProgress = $('pipeline-progress');
  const logSection = $('log-section');
  const logBody = $('log-body');
  const logCount = $('log-count');
  const reportSection = $('report-section');
  const reportBody = $('report-body');
  const btnDownload = $('btn-download');
  const headerStatus = $('header-status-text');
  const summaryGrid = $('summary-grid');

  // ─── Upload Handling ────────────────────────────
  uploadZone.addEventListener('click', () => fileInput.click());

  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });

  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
  });

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleFileSelect(fileInput.files[0]);
    }
  });

  function handleFileSelect(file) {
    state.file = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatSize(file.size);
    uploadZone.classList.add('hidden');
    configPanel.classList.add('visible');
    headerStatus.textContent = 'File loaded — configure analysis';
  }

  // ─── Analysis Launch ────────────────────────────
  btnAnalyze.addEventListener('click', async () => {
    if (!state.file) return;

    btnAnalyze.disabled = true;
    btnAnalyze.textContent = '⏳ Starting pipeline…';
    headerStatus.textContent = 'Launching pipeline…';

    const formData = new FormData();
    formData.append('file', state.file);
    formData.append('user_query', userQuery.value || '');
    formData.append('target_column', targetCol.value || '');

    try {
      const response = await fetch('/analyze', { method: 'POST', body: formData });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Analysis failed');
      }

      state.sessionId = data.session_id;
      state.pipelineStartTime = Date.now();
      state.processedEventIds = new Set();

      // Show pipeline sections immediately
      pipelineSection.classList.add('visible');
      logSection.classList.add('visible');

      // Optimistic: show ingestion as about to start
      setDAGNodeState('ingestion', 'running');
      addLog('Pipeline started — connecting to live stream…', 'phase-start');
      headerStatus.textContent = 'Pipeline running — ingestion';

      // Connect WebSocket for real-time events
      connectWebSocket(data.session_id);

      // Start polling fallback (in case WS misses events)
      startPolling(data.session_id);

    } catch (err) {
      btnAnalyze.disabled = false;
      btnAnalyze.textContent = '🚀 Launch DatacortexAI Analysis';
      addLog(`Error: ${err.message}`, 'error');
      headerStatus.textContent = 'Error — try again';
    }
  });

  // ─── Polling Fallback ──────────────────────────
  function startPolling(sessionId) {
    state.pollInterval = setInterval(async () => {
      try {
        const resp = await fetch(`/events/${sessionId}`);
        if (!resp.ok) return;
        const events = await resp.json();
        for (const event of events) {
          const eventKey = `${event.type}_${event.phase || ''}_${event.timestamp || ''}`;
          if (!state.processedEventIds.has(eventKey)) {
            state.processedEventIds.add(eventKey);
            handlePipelineEvent(event);
          }
        }
        // Stop polling if pipeline complete/failed
        if (events.some(e => e.type === 'pipeline_completed' || e.type === 'pipeline_failed')) {
          clearInterval(state.pollInterval);
        }
      } catch (e) { /* ignore polling errors */ }
    }, 3000);
  }

  // ─── WebSocket Connection ───────────────────────
  function connectWebSocket(sessionId) {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws/${sessionId}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      addLog('Connected to live pipeline stream', 'info');
    };

    ws.onmessage = (evt) => {
      try {
        const event = JSON.parse(evt.data);
        const eventKey = `${event.type}_${event.phase || ''}_${event.timestamp || ''}`;
        if (!state.processedEventIds.has(eventKey)) {
          state.processedEventIds.add(eventKey);
          handlePipelineEvent(event);
        }
      } catch (e) {
        console.warn('Failed to parse WS event:', e);
      }
    };

    ws.onerror = () => {
      addLog('WebSocket error — using polling fallback', 'error');
    };

    ws.onclose = () => {
      addLog('Stream disconnected', 'info');
    };

    state.ws = ws;
  }

  // ─── Pipeline Event Handling ────────────────────
  function handlePipelineEvent(event) {
    state.events.push(event);

    switch (event.type) {
      case 'phase_started':
        setDAGNodeState(event.phase, 'running');
        addLog(`Phase started: ${event.phase}`, 'phase-start');
        headerStatus.textContent = `Running: ${event.phase}`;
        break;

      case 'phase_completed':
        setDAGNodeState(event.phase, 'completed');
        addLog(`Phase completed: ${event.phase}`, 'phase-complete');
        updateProgress();
        if (event.result) {
          updateSummaryCards(event.phase, event.result);
        }
        break;

      case 'plan_generated':
        addLog(`Execution plan: ${(event.plan || []).join(' → ')}`, 'info');
        break;

      case 'pipeline_completed':
        headerStatus.textContent = 'Analysis complete ✓';
        pipelineProgress.textContent = '100%';
        btnAnalyze.textContent = '✓ Analysis Complete';
        if (event.report) {
          showReport(event.report);
        }
        const elapsed = event.elapsed_seconds || ((Date.now() - state.pipelineStartTime) / 1000);
        addLog(`Pipeline completed in ${elapsed.toFixed(1)}s`, 'phase-complete');
        break;

      case 'pipeline_failed':
        headerStatus.textContent = 'Pipeline failed ✗';
        addLog(`Pipeline failed: ${event.error}`, 'error');
        if (event.phase) {
          setDAGNodeState(event.phase, 'failed');
        }
        btnAnalyze.disabled = false;
        btnAnalyze.textContent = '🔄 Retry Analysis';
        break;

      default:
        if (event.message) {
          addLog(event.message, 'info');
        }
    }
  }

  // ─── DAG Node State ─────────────────────────────
  const phaseToNode = {
    ingestion: 'dag-ingestion',
    profiling: 'dag-profiling',
    eda: 'dag-eda',
    modeling: 'dag-modeling',
    reporting: 'dag-reporting',
  };

  const phaseToConnector = {
    ingestion: 'conn-1',
    profiling: 'conn-2',
    eda: 'conn-3',
    modeling: 'conn-4',
  };

  function setDAGNodeState(phase, state) {
    const nodeId = phaseToNode[phase];
    if (!nodeId) return;

    const node = $(nodeId);
    if (!node) return;

    node.className = `dag-node ${state}`;

    // Activate connector for completed phases
    if (state === 'completed') {
      const connId = phaseToConnector[phase];
      if (connId) {
        const conn = $(connId);
        if (conn) conn.classList.add('active');
      }
    }
  }

  function updateProgress() {
    const completed = document.querySelectorAll('.dag-node.completed').length;
    const total = document.querySelectorAll('.dag-node').length;
    const pct = Math.round((completed / total) * 100);
    pipelineProgress.textContent = `${pct}%`;
  }

  // ─── Summary Cards ──────────────────────────────
  function updateSummaryCards(phase, result) {
    if (phase === 'ingestion') {
      addSummaryCard(result.n_rows?.toLocaleString() || '—', 'Rows');
      addSummaryCard(result.n_cols || '—', 'Columns');
      if (result.quality?.overall_score) {
        addSummaryCard(`${(result.quality.overall_score * 100).toFixed(0)}%`, 'Data Quality');
      }
      if (result.fingerprint?.suggested_task) {
        addSummaryCard(result.fingerprint.suggested_task, 'Task Type');
      }
    }
    if (phase === 'eda' && result.findings) {
      addSummaryCard(result.findings.length, 'EDA Findings');
    }
    if (phase === 'modeling') {
      if (result.best_model) addSummaryCard(result.best_model, 'Best Model');
      if (result.best_score) addSummaryCard(`${(result.best_score * 100).toFixed(1)}%`, 'Best Score');
    }
  }

  function addSummaryCard(value, label) {
    const card = document.createElement('div');
    card.className = 'summary-card';
    card.innerHTML = `
      <div class="summary-value">${value}</div>
      <div class="summary-label">${label}</div>
    `;
    summaryGrid.appendChild(card);
  }

  // ─── Log Stream ─────────────────────────────────
  function addLog(message, type = 'info') {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;

    const now = new Date();
    const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;

    entry.innerHTML = `
      <span class="log-time">${timeStr}</span>
      <span class="log-msg">${escapeHtml(message)}</span>
    `;

    logBody.appendChild(entry);
    logBody.scrollTop = logBody.scrollHeight;
    logCount.textContent = `${logBody.children.length} events`;
  }

  // ─── Report Rendering ──────────────────────────
  function showReport(markdown) {
    state.reportMarkdown = markdown;
    reportSection.classList.add('visible');
    reportBody.innerHTML = renderMarkdown(markdown);
  }

  function renderMarkdown(md) {
    // Simple Markdown → HTML renderer
    let html = md;

    // Horizontal rules
    html = html.replace(/^---$/gm, '<hr>');

    // Headers
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Bold & Italic
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Tables
    html = renderTables(html);

    // Lists
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, (match) => {
      return `<ul>${match}</ul>`;
    });

    // Blockquotes
    html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');

    // Paragraphs
    html = html.replace(/\n\n/g, '</p><p>');
    html = `<p>${html}</p>`;

    // Clean up
    html = html.replace(/<p><\/p>/g, '');
    html = html.replace(/<p>(<h[123]>)/g, '$1');
    html = html.replace(/(<\/h[123]>)<\/p>/g, '$1');
    html = html.replace(/<p>(<ul>)/g, '$1');
    html = html.replace(/(<\/ul>)<\/p>/g, '$1');
    html = html.replace(/<p>(<hr>)<\/p>/g, '$1');
    html = html.replace(/<p>(<table)/g, '$1');
    html = html.replace(/(<\/table>)<\/p>/g, '$1');
    html = html.replace(/<p>(<blockquote>)/g, '$1');
    html = html.replace(/(<\/blockquote>)<\/p>/g, '$1');

    return html;
  }

  function renderTables(md) {
    const lines = md.split('\n');
    let result = [];
    let inTable = false;
    let tableRows = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (line.startsWith('|') && line.endsWith('|')) {
        if (!inTable) {
          inTable = true;
          tableRows = [];
        }
        // Skip separator rows
        if (/^\|[\s\-:|]+\|$/.test(line)) continue;
        tableRows.push(line);
      } else {
        if (inTable) {
          result.push(buildTable(tableRows));
          inTable = false;
          tableRows = [];
        }
        result.push(lines[i]);
      }
    }
    if (inTable) {
      result.push(buildTable(tableRows));
    }

    return result.join('\n');
  }

  function buildTable(rows) {
    if (rows.length === 0) return '';

    let html = '<table>';

    // Header
    const headerCells = rows[0].split('|').filter(c => c.trim());
    html += '<thead><tr>';
    headerCells.forEach(cell => html += `<th>${cell.trim()}</th>`);
    html += '</tr></thead>';

    // Body
    html += '<tbody>';
    for (let i = 1; i < rows.length; i++) {
      const cells = rows[i].split('|').filter(c => c.trim());
      html += '<tr>';
      cells.forEach(cell => html += `<td>${cell.trim()}</td>`);
      html += '</tr>';
    }
    html += '</tbody></table>';

    return html;
  }

  // ─── Download Report ────────────────────────────
  btnDownload.addEventListener('click', () => {
    if (!state.reportMarkdown) return;
    const blob = new Blob([state.reportMarkdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `datacortex_report_${state.sessionId || 'analysis'}.md`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // ─── Utilities ──────────────────────────────────
  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
})();
