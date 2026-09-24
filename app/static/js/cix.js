/**
 * CIX Platform — Core JavaScript & Final Enterprise Theme System
 * Handles Theme (Dark/Light), Sidebar state, Navigation active router, Toasts, Search, Upload, Tables
 */

// ── Theme System Module ─────────────────────────────────────────
const ThemeSystem = (() => {
  function getTheme() {
    return localStorage.getItem('cix-theme') || 'dark';
  }

  function setTheme(theme) {
    const isDark = theme === 'dark';
    document.documentElement.classList.toggle('dark', isDark);
    document.documentElement.classList.toggle('light', !isDark);
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('cix-theme', theme);

    // Update active Chart.js colors dynamically
    updateChartTheme(isDark);

    // Notify active page custom components
    window.dispatchEvent(new CustomEvent('cix-theme-change', { detail: { theme, isDark } }));
  }

  function toggleTheme() {
    const current = getTheme();
    const next = current === 'dark' ? 'light' : 'dark';
    setTheme(next);
    Toast.show(`Switched to ${next === 'dark' ? 'Dark' : 'Light'} Mode`, 'info', 2000);
  }

  function updateChartTheme(isDark) {
    if (typeof Chart === 'undefined') return;

    const textColor = isDark ? '#bcc9cd' : '#475569';
    const gridColor = isDark ? 'rgba(61,73,76,0.4)' : 'rgba(226,232,240,0.8)';
    const tooltipBg = isDark ? '#1d1f27' : '#ffffff';
    const tooltipText = isDark ? '#e1e2ec' : '#0f172a';
    const tooltipBorder = isDark ? '#3d494c' : '#cbd5e1';

    Chart.defaults.color = textColor;

    // Update all live Chart instances
    Object.values(Chart.instances || {}).forEach(chart => {
      if (!chart || !chart.options) return;
      if (chart.options.scales) {
        Object.values(chart.options.scales).forEach(scale => {
          if (scale.ticks) scale.ticks.color = textColor;
          if (scale.grid) scale.grid.color = gridColor;
        });
      }
      if (chart.options.plugins && chart.options.plugins.legend && chart.options.plugins.legend.labels) {
        chart.options.plugins.legend.labels.color = textColor;
      }
      if (chart.options.plugins && chart.options.plugins.tooltip) {
        chart.options.plugins.tooltip.backgroundColor = tooltipBg;
        chart.options.plugins.tooltip.titleColor = tooltipText;
        chart.options.plugins.tooltip.bodyColor = textColor;
        chart.options.plugins.tooltip.borderColor = tooltipBorder;
      }
      chart.update();
    });
  }

  function init() {
    const saved = getTheme();
    setTheme(saved);

    const toggleBtns = document.querySelectorAll('#themeToggleBtn, .theme-toggle-btn');
    toggleBtns.forEach(btn => btn.addEventListener('click', toggleTheme));
  }

  return { init, getTheme, setTheme, toggleTheme, updateChartTheme };
})();

// ── Toast System ──────────────────────────────────────────────
const Toast = (() => {
  let container;

  function init() {
    container = document.getElementById('toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'toast-container';
      document.body.appendChild(container);
    }
  }

  function show(message, type = 'info', duration = 4000) {
    if (!container) init();
    const icons = { success: 'check_circle', error: 'error', warning: 'warning', info: 'info' };
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
      <span class="material-symbols-outlined text-[20px] flex-shrink-0">${icons[type] || 'info'}</span>
      <span class="text-sm flex-1">${message}</span>
      <span class="toast-dismiss material-symbols-outlined text-[18px]" onclick="this.closest('.toast').remove()">close</span>
    `;
    container.appendChild(toast);
    if (duration > 0) setTimeout(() => toast.remove(), duration);
    return toast;
  }

  return { show, init };
})();

// ── Sidebar Management ────────────────────────────────────────
const Sidebar = (() => {
  let sidebar, collapseBtn, mobileToggle, overlay;
  let collapsed = false;

  function init() {
    sidebar = document.getElementById('sidebar');
    collapseBtn = document.getElementById('sidebarCollapseBtn');
    mobileToggle = document.getElementById('mobileSidebarToggle');
    overlay = document.getElementById('mobile-overlay');

    if (collapseBtn) collapseBtn.addEventListener('click', toggleCollapse);
    if (mobileToggle) mobileToggle.addEventListener('click', openMobile);
    if (overlay) overlay.addEventListener('click', closeMobile);

    // ESC key closes mobile sidebar drawer
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && sidebar && sidebar.classList.contains('mobile-open')) {
        closeMobile();
      }
    });

    // Close mobile drawer when clicking nav links
    if (sidebar) {
      sidebar.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
          if (window.innerWidth < 1024) closeMobile();
        });
      });
    }

    setActiveLink();
    restoreState();
  }

  function toggleCollapse() {
    collapsed = !collapsed;
    document.body.classList.toggle('sidebar-collapsed', collapsed);
    if (sidebar) sidebar.classList.toggle('collapsed', collapsed);
    localStorage.setItem('cix-sidebar-collapsed', collapsed);

    // Trigger window resize event after CSS transition finishes for Chart.js auto-reflow
    setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
    }, 360);
  }

  function openMobile() {
    if (!sidebar || !overlay) return;
    sidebar.classList.add('mobile-open');
    overlay.classList.add('active');
    document.body.style.overflow = 'hidden';
  }

  function closeMobile() {
    if (!sidebar || !overlay) return;
    sidebar.classList.remove('mobile-open');
    overlay.classList.remove('active');
    document.body.style.overflow = '';
  }

  function restoreState() {
    const saved = localStorage.getItem('cix-sidebar-collapsed');
    if (saved === 'true' && window.innerWidth >= 1024) {
      collapsed = true;
      document.body.classList.add('sidebar-collapsed');
      if (sidebar) sidebar.classList.add('collapsed');
      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
      }, 100);
    }
  }

  function setActiveLink() {
    const rawPath = window.location.pathname.replace(/\/$/, '') || '/';

    // Disambiguate customer list vs customer detail
    const isCustomerDetail = /^\/(customer|customers)\/.+$/.test(rawPath) || rawPath === '/customer';
    const isCustomersList = rawPath === '/customers';

    document.querySelectorAll('[data-path]').forEach(link => {
      const dp = link.dataset.path;
      const activeClasses = ['bg-surface-container-high', 'text-primary', 'font-bold', 'shadow-[inset_2px_0_0_0_#4cd7f6]'];
      const normalClasses = ['text-on-surface-variant', 'hover:bg-surface-container', 'hover:text-on-surface'];

      let isActive = false;

      if (dp === 'customer-360') {
        isActive = isCustomerDetail;
      } else if (dp === 'customers') {
        isActive = isCustomersList;
      } else {
        const pathMap = {
          overview: ['/dashboard', '/'],
          'data-upload': ['/upload'],
          'data-processing': ['/processing'],
          segments: ['/segments'],
          predictions: ['/predictions', '/churn'],
          recommendations: ['/recommendations'],
          analytics: ['/analytics'],
          reports: ['/reports'],
          explorer: ['/explorer'],
          settings: ['/settings'],
          admin: ['/admin'],
        };
        const paths = pathMap[dp] || [];
        isActive = paths.some(p => rawPath === p || rawPath.startsWith(p + '/'));
      }

      activeClasses.forEach(c => link.classList.remove(c));
      normalClasses.forEach(c => link.classList.remove(c));

      if (isActive) {
        link.setAttribute('aria-current', 'page');
        activeClasses.forEach(c => link.classList.add(c));
      } else {
        link.removeAttribute('aria-current');
        normalClasses.forEach(c => link.classList.add(c));
      }
    });
  }

  return { init, toggleCollapse, openMobile, closeMobile, setActiveLink };
})();

// ── Global Search ─────────────────────────────────────────────
const GlobalSearch = (() => {
  function init() {
    const input = document.getElementById('global-search');
    if (!input) return;

    // Cmd+K / Ctrl+K shortcut
    document.addEventListener('keydown', e => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        input.focus();
        input.select();
      }
    });

    input.addEventListener('keypress', e => {
      if (e.key === 'Enter') {
        const q = input.value.trim();
        if (q) window.location.href = `/customers?q=${encodeURIComponent(q)}`;
      }
    });
  }

  return { init };
})();

// ── File Upload Module ─────────────────────────────────────────
const Uploader = (() => {
  let dropZone, fileInput, progressWrap, progressBar, progressText, resultPanel;
  let selectedFile = null;

  function init() {
    dropZone     = document.getElementById('drop-zone');
    fileInput    = document.getElementById('file-input');
    progressWrap = document.getElementById('upload-progress-wrap');
    progressBar  = document.getElementById('upload-progress-bar');
    progressText = document.getElementById('upload-progress-text');
    resultPanel  = document.getElementById('upload-result-panel');

    if (!dropZone || !fileInput) return;

    // Drag and drop events
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', e => { if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove('drag-over'); });
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      const files = e.dataTransfer.files;
      if (files.length) handleFile(files[0]);
    });

    dropZone.addEventListener('click', e => {
      if (e.target.closest('button')) return;
      fileInput.click();
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length) handleFile(fileInput.files[0]);
    });

    const browseBtn = document.getElementById('browse-btn');
    if (browseBtn) browseBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });

    const uploadBtn = document.getElementById('upload-btn');
    if (uploadBtn) uploadBtn.addEventListener('click', performUpload);
  }

  function handleFile(file) {
    selectedFile = file;
    const allowed = ['.csv', '.xlsx', '.xls'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();

    if (!allowed.includes(ext)) {
      showFileInfo(file, 'error', 'Invalid format. Use CSV, XLSX or XLS.');
      selectedFile = null;
      return;
    }
    showFileInfo(file, 'ready');
  }

  function showFileInfo(file, state, error = '') {
    const infoEl = document.getElementById('file-info');
    if (!infoEl) return;

    const size = file.size > 1024 * 1024
      ? (file.size / (1024 * 1024)).toFixed(1) + ' MB'
      : (file.size / 1024).toFixed(0) + ' KB';

    const ext = file.name.split('.').pop().toLowerCase();
    const icon = 'table_chart';

    infoEl.classList.remove('hidden');
    infoEl.innerHTML = `
      <div class="flex items-center gap-3 p-4 rounded-xl bg-surface-container-high/60 border border-outline-variant/40 animate-fade-in-up">
        <span class="material-symbols-outlined text-primary text-2xl">${icon}</span>
        <div class="flex-1 min-w-0">
          <p class="text-sm font-medium text-on-surface truncate">${file.name}</p>
          <p class="text-xs text-on-surface-variant">${size} · ${ext.toUpperCase()}</p>
          ${error ? `<p class="text-xs text-error mt-1">${error}</p>` : ''}
        </div>
        ${state === 'ready' ? `<span class="material-symbols-outlined text-tertiary text-xl">check_circle</span>` : `<span class="material-symbols-outlined text-error text-xl">error</span>`}
      </div>
    `;

    const uploadBtn = document.getElementById('upload-btn');
    if (uploadBtn) uploadBtn.disabled = state !== 'ready';
  }

  function setProgress(pct, text = '') {
    if (progressWrap) progressWrap.classList.remove('hidden');
    if (progressBar) progressBar.style.width = pct + '%';
    if (progressText) progressText.textContent = text;
  }

  async function performUpload() {
    if (!selectedFile) { Toast.show('Please select a file first.', 'warning'); return; }

    const uploadBtn = document.getElementById('upload-btn');
    if (uploadBtn) { uploadBtn.disabled = true; uploadBtn.textContent = 'Processing...'; }

    setProgress(10, 'Uploading file...');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      setProgress(35, 'Validating schema & parsing features...');

      const resp = await fetch('/upload', { method: 'POST', body: formData });
      const data = await resp.json();

      setProgress(85, 'Running XGBoost propensity pipeline...');

      if (data.status === 'success') {
        setProgress(100, 'Pipeline Complete!');
        showUploadResult(data.upload_info);
        Toast.show(data.message, 'success');

        setTimeout(() => {
          if (data.redirect) window.location.href = data.redirect;
        }, 1200);
      } else {
        setProgress(0, '');
        if (progressWrap) progressWrap.classList.add('hidden');
        Toast.show(data.message || 'Upload failed.', 'error');
        if (uploadBtn) { uploadBtn.disabled = false; uploadBtn.textContent = 'Analyze Dataset'; }
      }
    } catch (e) {
      setProgress(0, '');
      if (progressWrap) progressWrap.classList.add('hidden');
      Toast.show('Network error: ' + e.message, 'error');
      if (uploadBtn) { uploadBtn.disabled = false; uploadBtn.textContent = 'Analyze Dataset'; }
    }
  }

  function showUploadResult(info) {
    if (!resultPanel) return;
    resultPanel.classList.remove('hidden');

    const statusColor = { success: 'text-tertiary', warning: 'text-warning', error: 'text-error' }[info.status] || 'text-tertiary';
    const statusIcon  = { success: 'check_circle', warning: 'warning', error: 'error' }[info.status] || 'check_circle';

    resultPanel.innerHTML = `
      <div class="rounded-xl bg-surface-container-low p-5 border border-outline-variant/30 animate-fade-in-up">
        <div class="flex items-center gap-3 mb-4">
          <span class="material-symbols-outlined ${statusColor} text-2xl">${statusIcon}</span>
          <div>
            <h3 class="font-semibold text-on-surface">Dataset Processed Successfully</h3>
            <p class="text-xs text-on-surface-variant">${info.filename}</p>
          </div>
        </div>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <div class="bg-surface-container p-3 rounded-lg text-center">
            <div class="text-lg font-bold text-primary font-mono">${(info.rows || 0).toLocaleString()}</div>
            <div class="text-xs text-on-surface-variant">Rows</div>
          </div>
          <div class="bg-surface-container p-3 rounded-lg text-center">
            <div class="text-lg font-bold text-secondary font-mono">${info.columns || 0}</div>
            <div class="text-xs text-on-surface-variant">Columns</div>
          </div>
          <div class="bg-surface-container p-3 rounded-lg text-center">
            <div class="text-lg font-bold ${info.missing_cells > 0 ? 'text-warning' : 'text-tertiary'} font-mono">${(info.missing_cells || 0).toLocaleString()}</div>
            <div class="text-xs text-on-surface-variant">Missing Values</div>
          </div>
          <div class="bg-surface-container p-3 rounded-lg text-center">
            <div class="text-lg font-bold ${info.duplicate_rows > 0 ? 'text-warning' : 'text-tertiary'} font-mono">${info.duplicate_rows || 0}</div>
            <div class="text-xs text-on-surface-variant">Duplicates</div>
          </div>
        </div>
        <div class="text-xs text-on-surface-variant">
          ML pipeline complete. Redirecting to Data Processing view...
        </div>
      </div>
    `;
  }

  return { init };
})();

// ── Confirmation Modal ────────────────────────────────────────
const Modal = (() => {
  function confirm(title, body, onConfirm, danger = false) {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.innerHTML = `
      <div class="modal-box" role="dialog" aria-modal="true" aria-label="${title}">
        <h2 class="text-lg font-semibold text-on-surface mb-2">${title}</h2>
        <p class="text-sm text-on-surface-variant mb-6">${body}</p>
        <div class="flex items-center justify-end gap-3">
          <button id="modal-cancel" class="px-4 py-2 rounded-lg bg-surface-container text-on-surface-variant hover:text-on-surface text-sm transition-colors">Cancel</button>
          <button id="modal-confirm" class="px-4 py-2 rounded-lg ${danger ? 'bg-error text-white hover:bg-error/80' : 'bg-primary text-on-primary hover:bg-primary/90'} text-sm font-semibold transition-all">Confirm</button>
        </div>
      </div>
    `;

    document.body.appendChild(overlay);
    overlay.querySelector('#modal-cancel').onclick = () => overlay.remove();
    overlay.querySelector('#modal-confirm').onclick = () => { overlay.remove(); onConfirm(); };
    overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
    overlay.querySelector('#modal-confirm').focus();
  }

  return { confirm };
})();

// ── Data Actions ──────────────────────────────────────────────
function clearDataset() {
  Modal.confirm(
    'Clear Current Dataset',
    'This will remove the uploaded dataset from memory and restore default state. This action cannot be undone.',
    async () => {
      try {
        const resp = await fetch('/api/clear', { method: 'POST' });
        const data = await resp.json();
        if (data.status === 'success') {
          Toast.show('Dataset cleared successfully.', 'success');
          setTimeout(() => window.location.href = '/upload', 1200);
        }
      } catch (e) {
        Toast.show('Failed to clear dataset: ' + e.message, 'error');
      }
    },
    true
  );
}

// ── Export Helpers ────────────────────────────────────────────
function exportCSV(type = 'intelligence') {
  Toast.show('Preparing CSV export...', 'info', 2000);
  window.location.href = `/export/csv?type=${type}`;
}

function exportExcel(type = 'intelligence') {
  Toast.show('Preparing Excel export...', 'info', 2000);
  window.location.href = `/export/excel?type=${type}`;
}

// ── Table Utilities ───────────────────────────────────────────
const CIXTable = (() => {
  function renderPagination(container, page, totalPages, onPage) {
    if (!container) return;
    container.innerHTML = '';

    const maxButtons = 5;
    let start = Math.max(1, page - Math.floor(maxButtons / 2));
    let end = Math.min(totalPages, start + maxButtons - 1);
    if (end - start < maxButtons - 1) start = Math.max(1, end - maxButtons + 1);

    const btn = (label, pg, disabled = false, active = false) => {
      const el = document.createElement('button');
      el.className = `px-3 py-1.5 rounded text-xs font-medium transition-all ${
        active ? 'bg-primary text-on-primary font-bold shadow-[0_0_12px_rgba(76,215,246,0.3)]' :
        disabled ? 'text-outline opacity-40 cursor-not-allowed' :
        'text-on-surface-variant hover:bg-surface-container hover:text-on-surface'
      }`;
      el.textContent = label;
      if (!disabled && !active) el.onclick = () => onPage(pg);
      return el;
    };

    container.appendChild(btn('‹', page - 1, page <= 1));
    for (let p = start; p <= end; p++) container.appendChild(btn(p, p, false, p === page));
    container.appendChild(btn('›', page + 1, page >= totalPages));
  }

  function badgeClass(value) {
    if (!value || value === '—') return 'bg-surface-container text-on-surface-variant';
    const v = value.toString().toLowerCase();
    if (v.includes('champion')) return 'badge-Champions';
    if (v.includes('loyal') && !v.includes('potential')) return 'badge-Loyal';
    if (v.includes('potential')) return 'badge-Potential';
    if (v.includes('at risk') || v.includes('at-risk')) return 'badge-AtRisk';
    if (v.includes('needs')) return 'badge-Needs';
    if (v.includes('hibernat')) return 'badge-Hibernating';
    if (v.includes('new') || v.includes('promis')) return 'badge-New';
    if (v === 'high') return 'badge-High';
    if (v === 'medium') return 'badge-Medium';
    if (v === 'low') return 'badge-Low';
    return 'bg-surface-container-high text-on-surface-variant';
  }

  return { renderPagination, badgeClass };
})();

// ── Init ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  Toast.init();
  ThemeSystem.init();
  Sidebar.init();
  GlobalSearch.init();
  Uploader.init();

  // Handle flash messages passed from Flask
  const flashMessages = document.querySelectorAll('[data-flash]');
  flashMessages.forEach(el => {
    Toast.show(el.textContent.trim(), el.dataset.flashType || 'info');
  });
});
