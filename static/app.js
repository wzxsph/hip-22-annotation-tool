const DEFAULT_LANDMARKS = [
  { number: 1, name: "acetabular_outer", label_zh: "髋臼外上缘" },
  { number: 2, name: "triradiate_center", label_zh: "Y 形软骨中心" },
  { number: 3, name: "femoral_head_center", label_zh: "股骨头中心" },
  { number: 4, name: "teardrop_lower", label_zh: "泪滴下缘" },
  { number: 5, name: "femoral_shaft_prox", label_zh: "股骨干轴近端中心" },
  { number: 6, name: "femoral_shaft_dist", label_zh: "股骨干远端中心" },
  { number: 7, name: "femoral_neck_axis_center", label_zh: "股骨颈轴中心" },
  { number: 8, name: "femoral_head_medial", label_zh: "股骨头最内侧缘" },
  { number: 9, name: "femoral_head_lateral", label_zh: "股骨头最外侧缘" },
  { number: 10, name: "obturator_upper", label_zh: "闭孔上缘" },
  { number: 11, name: "femoral_neck_inner_lower", label_zh: "股骨颈内下缘" },
];

const DEFAULT_SETTINGS = {
  dataset_root: "annotation-tool",
  auto_detect: true,
  autosave: true,
  annotator: "default",
};

const SCAN_CORNER_LABELS = ["左上", "右上", "右下", "左下"];
const DEFAULT_HIDDEN_POINT_NUMBERS = new Set([10, 11]);

const App = {
  state: {
    schema: { sides: ["left", "right"], landmarks: DEFAULT_LANDMARKS },
    settings: { ...DEFAULT_SETTINGS },
    image: null,
    imageBaseUrl: "",
    imageView: "enhanced",
    annotation: null,
    currentFilename: null,
    transform: { x: 0, y: 0, scale: 1 },
    activeTool: "select",
    activePointKey: "left_acetabular_outer",
    selectedPoint: null,
    selectedShentonPoint: null,
    selectedScanCorner: null,
    selectedConnection: null,
    pendingConnectionStart: null,
    isDragging: false,
    isDraggingShenton: false,
    isDraggingScanCorner: false,
    isDrawingRoi: false,
    isPanning: false,
    dragStarted: false,
    spaceHeld: false,
    showLabels: true,
    showDefaultConnections: true,
    showManualConnections: true,
    showShenton: true,
    showMeasurements: true,
    showPoint10And11: false,
    shentonSide: "left",
    shentonSegment: "obturator_upper_curve",
    roiStart: null,
    roiDraft: null,
    lastMouse: { x: 0, y: 0 },
    contextImagePos: { x: 0, y: 0 },
    history: [],
    historyIndex: -1,
    manifestImages: [],
    progress: null,
    statusFilter: "all",
    importReport: null,
    lastSave: null,
    lastSavedSnapshot: "",
    manifestByFilename: new Map(),
    thumbByFilename: new Map(),
    autosaveTimer: null,
    measurementTimer: null,
    thumbObserver: null,
    autoDetectPollTimer: null,
    autoDetectStatus: null,
    isNavigating: false,
  },

  init: async () => {
    await App.loadSettings();
    await App.loadSchema();
    App.buildPointList();
    App.bindEvents();
    App.setTool("select");
    App.updateSettingsPanel();
    App.resize();
    window.addEventListener("resize", App.resize);
    requestAnimationFrame(App.drawLoop);
    await App.loadManifest();
    App.startAutoDetectPolling();
  },

  loadSettings: async () => {
    try {
      const res = await fetch("/api/annotation/settings");
      if (res.ok) {
        App.state.settings = { ...DEFAULT_SETTINGS, ...(await App.readJsonResponse(res, "settings failed")) };
      }
    } catch (err) {
      console.warn("Settings fallback", err);
    }
  },

  saveSettings: async () => {
    const payload = {
      auto_detect: document.getElementById("autoDetectToggle").checked,
      autosave: document.getElementById("autoSaveToggle").checked,
      annotator: document.getElementById("annotatorInput").value.trim() || "default",
    };
    try {
      const res = await fetch("/api/annotation/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (res.ok) App.state.settings = { ...DEFAULT_SETTINGS, ...(await App.readJsonResponse(res, "settings failed")) };
      App.updateSettingsPanel();
    } catch (err) {
      App.setStatus(`设置保存失败: ${err.message}`);
    }
  },

  updateSettingsPanel: () => {
    document.getElementById("autoDetectToggle").checked = Boolean(App.state.settings.auto_detect);
    document.getElementById("autoSaveToggle").checked = Boolean(App.state.settings.autosave);
    document.getElementById("datasetRootText").textContent = App.state.settings.dataset_root || "annotation-tool";
    App.renderAutoDetectStatus();
    if (!App.state.annotation) {
      document.getElementById("annotatorInput").value = App.state.settings.annotator || "default";
    }
  },

  loadSchema: async () => {
    try {
      const res = await fetch("/api/annotation/schema");
      if (res.ok) App.state.schema = await App.readJsonResponse(res, "schema failed");
    } catch (err) {
      console.warn("Schema fallback", err);
    }
  },

  bindEvents: () => {
    document.getElementById("fileInput").addEventListener("change", App.handleFileSelect);
    document.getElementById("btnFolder").addEventListener("click", () => document.getElementById("folderDialog").showModal());
    document.getElementById("btnPickFolder").addEventListener("click", App.handlePickFolder);
    document.getElementById("btnConfirmFolder").addEventListener("click", App.handleOpenFolder);
    document.getElementById("btnChooseDatasetRoot").addEventListener("click", App.handleChooseDatasetRoot);
    document.getElementById("autoDetectToggle").addEventListener("change", App.saveSettings);
    document.getElementById("autoSaveToggle").addEventListener("change", App.saveSettings);
    document.getElementById("annotatorInput").addEventListener("change", App.saveSettings);
    document.getElementById("btnSave").addEventListener("click", () => App.saveAnnotation());
    document.getElementById("btnClearPoints").addEventListener("click", App.clearCurrentImagePoints);
    document.getElementById("btnDeleteImage").addEventListener("click", App.deleteCurrentImage);
    document.getElementById("btnConfirmKeypointsComplete").addEventListener("click", App.confirmKeypointsComplete);
    document.getElementById("btnConfirmShentonComplete").addEventListener("click", App.confirmShentonComplete);
    document.getElementById("btnAutoDetect").addEventListener("click", App.handleAutoDetectCurrent);
    document.getElementById("btnAutoDetectEnhanced").addEventListener("click", () => App.handleAutoDetectCurrent({ useEnhanced: true }));
    document.getElementById("btnShortcutHelp").addEventListener("click", App.showShortcuts);
    document.getElementById("btnExportYolo").addEventListener("click", () => {
      window.location.href = "/api/annotation/export-yolo";
    });
    document.getElementById("btnUndo").addEventListener("click", App.undo);
    document.getElementById("btnRedo").addEventListener("click", App.redo);
    document.getElementById("btnFit").addEventListener("click", App.fitToScreen);
    document.getElementById("btnToggleLabels").addEventListener("click", () => {
      App.state.showLabels = !App.state.showLabels;
      App.syncDisplayToggles();
    });
    document.getElementById("btnDeleteConnection").addEventListener("click", App.deleteSelectedConnection);
    document.getElementById("showDefaultConnectionsToggle").addEventListener("change", (event) => {
      App.state.showDefaultConnections = event.target.checked;
      App.renderConnectionList();
    });
    document.getElementById("showManualConnectionsToggle").addEventListener("change", (event) => {
      App.state.showManualConnections = event.target.checked;
      App.renderConnectionList();
    });
    document.getElementById("showShentonToggle").addEventListener("change", (event) => {
      App.state.showShenton = event.target.checked;
    });
    document.getElementById("showMeasurementsToggle").addEventListener("change", (event) => {
      App.state.showMeasurements = event.target.checked;
      App.renderMeasurements();
    });
    document.getElementById("showPointLabelsToggle").addEventListener("change", (event) => {
      App.state.showLabels = event.target.checked;
    });
    document.getElementById("showPoint10And11Toggle").addEventListener("change", (event) => {
      App.state.showPoint10And11 = event.target.checked;
    });
    document.querySelectorAll("[data-image-view]").forEach((button) => {
      button.addEventListener("click", () => App.setImageView(button.dataset.imageView || "original"));
    });
    document.querySelectorAll("[data-shenton-side]").forEach((button) => {
      button.addEventListener("click", () => App.setShentonSide(button.dataset.shentonSide || "left"));
    });
    document.querySelectorAll("[data-shenton-segment]").forEach((button) => {
      button.addEventListener("click", () => App.setShentonSegment(button.dataset.shentonSegment || "obturator_upper_curve"));
    });
    document.getElementById("btnShentonUndo").addEventListener("click", App.undoShentonPoint);
    document.getElementById("btnShentonClear").addEventListener("click", App.clearShentonSegment);
    document.getElementById("btnRoiClear").addEventListener("click", App.clearRoiCrop);
    document.getElementById("btnRoiFit").addEventListener("click", App.fitToRoi);
    document.getElementById("btnScanClear").addEventListener("click", App.clearScanTransform);
    document.getElementById("btnScanFit").addEventListener("click", App.fitToScanTransform);
    document.getElementById("shentonReviewSelect").addEventListener("change", App.updateShentonReview);
    document.querySelectorAll("#progressFilters button").forEach((button) => {
      button.addEventListener("click", () => App.setStatusFilter(button.dataset.statusFilter || "all"));
    });

    document.querySelectorAll(".tool-button").forEach((button) => {
      button.addEventListener("click", () => App.setTool(button.dataset.tool));
    });

    const canvas = document.getElementById("mainCanvas");
    canvas.addEventListener("mousedown", App.handleMouseDown);
    canvas.addEventListener("mousemove", App.handleMouseMove);
    canvas.addEventListener("mouseup", App.handleMouseUp);
    canvas.addEventListener("mouseleave", App.handleMouseUp);
    canvas.addEventListener("wheel", App.handleWheel, { passive: false });
    canvas.addEventListener("contextmenu", App.handleContextMenu);

    document.addEventListener("keydown", App.handleKeyDown);
    document.addEventListener("keyup", App.handleKeyUp);
    document.addEventListener("click", (event) => {
      if (!event.target.closest("#contextMenu")) App.hideContextMenu();
    });
  },

  resize: () => {
    const canvas = document.getElementById("mainCanvas");
    const rect = canvas.parentElement.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * ratio));
    canvas.height = Math.max(1, Math.floor(rect.height * ratio));
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  },

  showShortcuts: () => {
    const dialog = document.getElementById("shortcutsDialog");
    if (dialog?.showModal) dialog.showModal();
  },

  buildPointList: () => {
    const container = document.getElementById("pointGroups");
    container.innerHTML = "";
    const labels = { left: "图像左侧", right: "图像右侧" };
    App.state.schema.sides.forEach((side) => {
      const group = document.createElement("section");
      group.className = "point-group";
      const title = document.createElement("h3");
      title.textContent = labels[side] || side;
      group.appendChild(title);
      App.state.schema.landmarks.forEach((landmark) => {
        const key = App.keyFor(side, landmark.name);
        const row = document.createElement("div");
        row.className = "point-row missing";
        row.dataset.key = key;
        row.innerHTML = `
          <span class="point-number">#${landmark.number}</span>
          <input type="checkbox" aria-label="visible" />
          <span class="point-name">${landmark.label_zh}</span>
          <span class="source-badge missing">缺失</span>
        `;
        row.addEventListener("click", (event) => {
          if (event.target.tagName !== "INPUT") {
            App.selectPoint(key);
            App.state.activePointKey = key;
          }
        });
        row.querySelector("input").addEventListener("change", (event) => {
          event.stopPropagation();
          App.togglePointVisibility(key, event.target.checked);
        });
        group.appendChild(row);
      });
      container.appendChild(group);
    });
  },

  keyFor: (side, name) => `${side}_${name}`,

  setTool: (tool) => {
    App.state.activeTool = tool;
    if (tool !== "line") App.state.pendingConnectionStart = null;
    document.querySelectorAll(".tool-button").forEach((button) => {
      button.classList.toggle("active", button.dataset.tool === tool);
    });
    const shentonToolbox = document.getElementById("shentonToolbox");
    const roiToolbox = document.getElementById("roiToolbox");
    const scanToolbox = document.getElementById("scanToolbox");
    if (shentonToolbox) shentonToolbox.hidden = tool !== "shenton";
    if (roiToolbox) roiToolbox.hidden = tool !== "roi";
    if (scanToolbox) scanToolbox.hidden = tool !== "scan";
    document.getElementById("mainCanvas").style.cursor =
      tool === "point" || tool === "line" || tool === "shenton" || tool === "roi" || tool === "scan" ? "crosshair" : "default";
    App.updateSelectedBox();
  },

  setStatus: (message) => {
    document.getElementById("statusText").textContent = message;
  },

  readJsonResponse: async (res, fallbackMessage = "请求失败") => {
    const text = await res.text();
    let payload = {};
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch (err) {
        const snippet = text.slice(0, 220).trim();
        throw new Error(snippet || fallbackMessage);
      }
    }
    if (!res.ok) {
      throw new Error(payload.detail || payload.message || fallbackMessage);
    }
    return payload;
  },

  handlePickFolder: async () => {
    const error = document.getElementById("folderError");
    error.textContent = "";
    try {
      const res = await fetch("/api/annotation/select-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ purpose: "import" }),
      });
      const data = await App.readJsonResponse(res, "folder picker failed");
      document.getElementById("folderPathInput").value = data.path || "";
    } catch (err) {
      error.textContent = `${err.message}；可手动填写路径。`;
    }
  },

  handleChooseDatasetRoot: async () => {
    App.setStatus("请选择保存目录...");
    try {
      const res = await fetch("/api/annotation/select-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ purpose: "dataset" }),
      });
      const data = await App.readJsonResponse(res, "folder picker failed");
      App.state.settings = { ...DEFAULT_SETTINGS, ...(data.settings || {}) };
      App.updateSettingsPanel();
      App.clearWorkspaceView("工作区已切换");
      await App.loadManifest();
      App.startAutoDetectPolling();
      App.setStatus("保存目录已更新");
    } catch (err) {
      App.setStatus(`选择保存目录失败: ${err.message}`);
    }
  },

  handleFileSelect: async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    await App.saveSettings();
    const form = new FormData();
    form.append("file", file);
    App.setStatus("正在载入...");
    try {
      const res = await fetch("/api/annotation/load", { method: "POST", body: form });
      App.applyAnnotation(await App.readJsonResponse(res, "load failed"));
      await App.loadManifest();
    } catch (err) {
      App.setStatus(`载入失败: ${err.message}`);
    } finally {
      event.target.value = "";
    }
  },

  handleOpenFolder: async () => {
    await App.saveSettings();
    const dialog = document.getElementById("folderDialog");
    const path = document.getElementById("folderPathInput").value.trim();
    const error = document.getElementById("folderError");
    error.textContent = "";
    try {
      const res = await fetch("/api/annotation/open-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder_path: path, split: "train" }),
      });
      const data = await App.readJsonResponse(res, "import failed");
      App.state.settings = { ...DEFAULT_SETTINGS, ...(data.settings || {}) };
      App.state.autoDetectStatus = data.auto_detect || null;
      App.state.importReport = data.import_report || null;
      App.clearWorkspaceView("文件夹已导入，自动识别将在后台进行");
      App.state.importReport = data.import_report || null;
      dialog.close();
      await App.loadManifest();
      App.startAutoDetectPolling();
      App.renderImportReport();
      App.setStatus(`文件夹已导入：${data.total || 0} 张，后台识别 ${data.queued || 0} 张`);
    } catch (err) {
      error.textContent = err.message;
    }
  },

  handleAutoDetectCurrent: async (options = {}) => {
    if (!App.state.currentFilename || !App.state.annotation) {
      App.setStatus("请先打开一张图片");
      return;
    }
    const useEnhanced = Boolean(options.useEnhanced);
    const button = document.getElementById(useEnhanced ? "btnAutoDetectEnhanced" : "btnAutoDetect");
    button.disabled = true;
    const saved = await App.saveAnnotation({ silent: true, skipManifest: true });
    if (saved === false) {
      button.disabled = false;
      return;
    }
    const useRoi = Boolean(App.currentRoiCrop());
    const useScan = !useRoi;
    const requestedLabel = useRoi ? (useEnhanced ? "ROI内增强识别" : "ROI内原图识别") : useEnhanced ? "增强识别" : "原图识别";
    App.setStatus(`正在${requestedLabel}...`);
    try {
      const res = await fetch("/api/annotation/auto-detect-image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: App.state.currentFilename,
          preserve_manual: true,
          include_partial: true,
          use_enhanced: useEnhanced,
          use_roi: useRoi,
          use_scan: useScan,
        }),
      });
      const data = await App.readJsonResponse(res, "auto detect image failed");
      const info = data.auto_detect || {};
      const resultLabel = App.autoDetectResultLabel(info, useEnhanced);
      App.applyAnnotation(data);
      App.state.lastSave = {
        filename: data.image.filename,
        timeText: new Date().toLocaleTimeString(),
        annotationPath: info.annotation_path || "annotations",
        labelPath: info.label_path || "同名 txt",
      };
      await App.loadManifest();
      App.renderSaveInfo();
      if (info.applied && info.visible_count > 0) {
        App.setStatus(`${resultLabel}完成：识别到 ${info.visible_count} 个点，未识别到的点保持缺失`);
      } else {
        App.setStatus(`${resultLabel}未找到可用点，所有未识别点保持缺失，请人工标注或把图片发给项目团队排查`);
      }
    } catch (err) {
      App.setStatus(`识别失败: ${err.message}`);
    } finally {
      button.disabled = false;
    }
  },

  autoDetectResultLabel: (info, useEnhanced = false) => {
    if (info?.roi_crop_used) return useEnhanced ? "ROI内增强识别" : "ROI内原图识别";
    if (info?.scan_transform_used) return "扫描校正识别";
    return useEnhanced ? "全图增强识别" : "全图识别";
  },

  loadManifest: async () => {
    try {
      const res = await fetch("/api/annotation/list");
      if (!res.ok) throw new Error(await res.text());
      const data = await App.readJsonResponse(res, "manifest failed");
      App.state.manifestImages = data.images || [];
      App.state.manifestByFilename = new Map(App.state.manifestImages.map((item) => [App.imageFilename(item), item]));
      App.state.progress = data.progress || null;
      App.state.autoDetectStatus = data.auto_detect || App.state.autoDetectStatus;
      if (data.settings) {
        App.state.settings = { ...DEFAULT_SETTINGS, ...data.settings };
        App.updateSettingsPanel();
      }
      App.renderThumbnails();
      App.renderAutoDetectStatus();
      App.renderProgressSummary();
    } catch (err) {
      console.warn(err);
    }
  },

  setStatusFilter: (filter) => {
    App.state.statusFilter = filter || "all";
    document.querySelectorAll("#progressFilters button").forEach((button) => {
      button.classList.toggle("active", button.dataset.statusFilter === App.state.statusFilter);
    });
    App.renderThumbnails();
  },

  progressCounts: () => {
    const counts = {
      total: 0,
      pending: 0,
      auto: 0,
      in_progress: 0,
      keypoint_complete: 0,
      shenton_awaiting_confirmation: 0,
      shenton_complete: 0,
      done: 0,
      needs_review: 0,
      keypoints_complete_total: 0,
      shenton_complete_total: 0,
    };
    (App.state.manifestImages || []).forEach((item) => {
      const status = ["pending", "auto", "in_progress", "keypoint_complete", "shenton_awaiting_confirmation", "shenton_complete", "done"].includes(item.status)
        ? item.status
        : "pending";
      counts[status] = (counts[status] || 0) + 1;
      counts.total += 1;
      if (item.keypoint_status === "complete") counts.keypoints_complete_total += 1;
      if (item.shenton_status === "complete") counts.shenton_complete_total += 1;
    });
    counts.needs_review = counts.total - counts.done;
    return counts;
  },

  renderProgressSummary: () => {
    const counts = { ...App.progressCounts(), ...(App.state.progress?.counts || {}) };
    const summary = document.getElementById("progressSummary");
    if (!summary) return;
    if (!counts.total) {
      summary.textContent = "未导入文件夹";
    } else {
      summary.innerHTML = `<strong>${counts.done}</strong> / ${counts.total} 完成；${counts.needs_review} 张仍需处理<br><span>关键点完成 ${counts.keypoint_complete || 0}；Shenton完成 ${counts.shenton_complete || 0}；未完成 ${counts.in_progress || 0}</span>`;
    }
    const report = App.state.progress?.report_files;
    const reportText = document.getElementById("progressReportText");
    if (reportText && report) {
      reportText.textContent = `已生成 ${report.marker}、${report.html} 和 ${report.csv}`;
    }
    document.querySelectorAll("#progressFilters button").forEach((button) => {
      const filter = button.dataset.statusFilter || "all";
      const count = filter === "all" ? counts.total : counts[filter] || 0;
      button.textContent = `${App.filterLabel(filter)} ${count}`;
      button.classList.toggle("active", filter === App.state.statusFilter);
    });
    App.renderSubmissionCheck(counts);
    App.renderSaveInfo();
    App.renderImportReport();
  },

  renderSubmissionCheck: (counts = null) => {
    const target = document.getElementById("submissionCheck");
    if (!target) return;
    const nextCounts = counts || App.state.progress?.counts || App.progressCounts();
    target.classList.remove("ready", "warning");
    if (!nextCounts.total) {
      target.textContent = "导入文件夹后会显示提交检查";
      return;
    }
    if (nextCounts.needs_review === 0) {
      target.classList.add("ready");
      target.textContent = "提交检查通过：全部图片关键点和 Shenton 均已完成。保存后将整个文件夹直接发送给项目团队。";
      return;
    }
    target.classList.add("warning");
    target.textContent = `提交前仍需处理 ${nextCounts.needs_review} 张：未标注 ${nextCounts.pending}，待复核 ${nextCounts.auto}，未完成 ${nextCounts.in_progress}，关键点完成 ${nextCounts.keypoint_complete || 0}，Shenton完成 ${nextCounts.shenton_complete || 0}。`;
  },

  renderSaveInfo: () => {
    const target = document.getElementById("saveInfoText");
    if (!target) return;
    if (!App.state.currentFilename) {
      target.textContent = "尚未打开当前图片";
      return;
    }
    if (!App.state.lastSave || App.state.lastSave.filename !== App.state.currentFilename) {
      target.textContent = "当前图片修改后请保存；保存会写入 JSON 和同名 YOLO txt";
      return;
    }
    target.textContent = `最近保存 ${App.state.lastSave.timeText}；JSON：${App.state.lastSave.annotationPath}；TXT：${App.state.lastSave.labelPath}`;
  },

  renderImportReport: () => {
    const target = document.getElementById("importReportText");
    if (!target) return;
    const report = App.state.importReport;
    const parts = [];
    if (report?.warnings?.length) parts.push(...report.warnings.slice(0, 3));
    if (report?.legacy_annotations) parts.push(`旧标注 ${report.legacy_annotations} 个`);
    if (report?.copied_external_images) parts.push(`已复制图片 ${report.copied_external_images} 张`);
    if (report?.missing_legacy_images?.length) parts.push(`缺图 ${report.missing_legacy_images.length} 个`);
    if (report?.conflicting_legacy_images?.length) parts.push(`冲突 ${report.conflicting_legacy_images.length} 个`);
    if (!parts.length) {
      target.textContent = "命名混乱或目录嵌套时，可先发给项目团队整理";
      return;
    }
    target.textContent = `导入提示：${parts.slice(0, 5).join("；")}`;
  },

  renderThumbnails: () => {
    const strip = document.getElementById("thumbStrip");
    strip.innerHTML = "";
    App.state.thumbByFilename = new Map();
    if (App.state.thumbObserver) {
      App.state.thumbObserver.disconnect();
      App.state.thumbObserver = null;
    }
    const supportsLazyThumbs = "IntersectionObserver" in window;
    if (supportsLazyThumbs) {
      App.state.thumbObserver = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            App.loadThumbnail(entry.target);
            App.state.thumbObserver.unobserve(entry.target);
          });
        },
        { root: strip, rootMargin: "360px" },
      );
    }
    const images = App.filteredManifestImages();
    if (!images.length) {
      const empty = document.createElement("div");
      empty.className = "thumb-empty";
      empty.textContent = App.state.manifestImages.length ? "当前筛选没有图片" : "导入文件夹后会显示图片状态";
      strip.appendChild(empty);
      return;
    }
    images.forEach((item, index) => {
      const filename = item.image_path.split("/").pop();
      const thumb = document.createElement("button");
      thumb.type = "button";
      thumb.className = `thumb loading ${filename === App.state.currentFilename ? "active" : ""}`;
      thumb.dataset.filename = filename;
      thumb.title = App.manifestStatusTitle(item);
      thumb.innerHTML = `
        <span class="thumb-status ${item.status || "pending"}"></span>
        <span class="thumb-label">${App.statusShortLabel(item.status)}</span>
      `;
      thumb.addEventListener("click", () => App.loadByName(filename));
      strip.appendChild(thumb);
      App.state.thumbByFilename.set(filename, thumb);
      if (App.state.thumbObserver) {
        App.state.thumbObserver.observe(thumb);
      } else if (index < 80) {
        App.loadThumbnail(thumb);
      }
    });
  },

  filteredManifestImages: () => {
    const filter = App.state.statusFilter || "all";
    if (filter === "all") return App.state.manifestImages || [];
    return (App.state.manifestImages || []).filter((item) => (item.status || "pending") === filter);
  },

  loadThumbnail: (thumb) => {
    if (!thumb || thumb.dataset.loaded === "1") return;
    const filename = thumb.dataset.filename;
    thumb.style.backgroundImage = `url(/api/annotation/image/${encodeURIComponent(filename)}?thumb=1)`;
    thumb.dataset.loaded = "1";
    thumb.classList.remove("loading");
  },

  setActiveThumbnail: (filename) => {
    App.state.thumbByFilename.forEach((thumb, key) => {
      thumb.classList.toggle("active", key === filename);
    });
  },

  updateManifestEntryStatus: (filename, progressOrStatus) => {
    const progress = typeof progressOrStatus === "string" ? { status: progressOrStatus } : progressOrStatus || {};
    const status = progress.status;
    if (!filename || !status) { console.log("updateManifestEntryStatus: missing filename or status", { filename, status }); return; }
    const item = App.state.manifestByFilename.get(filename);
    if (!item) { console.log("updateManifestEntryStatus: item not found for", filename, "keys:", [...App.state.manifestByFilename.keys()]); return; }
    item.status = status;
    item.keypoint_status = progress.keypoint_status || item.keypoint_status || "pending";
    item.shenton_status = progress.shenton_status || item.shenton_status || "pending";
    item.status_detail = progress.status_detail || item.status_detail || "";
    if (progress.keypoints) {
      item.keypoint_visible_count = progress.keypoints.visible;
      item.keypoint_manual_count = progress.keypoints.manual;
    }
    if (progress.shenton) {
      item.shenton_complete_sides = progress.shenton.complete_sides;
      item.shenton_started_sides = progress.shenton.started_sides;
    }
    const thumb = App.state.thumbByFilename.get(filename);
    if (thumb) {
      const statusDot = thumb.querySelector(".thumb-status");
      const label = thumb.querySelector(".thumb-label");
      if (statusDot) statusDot.className = `thumb-status ${status}`;
      if (label) label.textContent = App.statusShortLabel(status);
      thumb.title = App.manifestStatusTitle(item);
    }
  },

  manifestStatusTitle: (item) => {
    const filename = App.imageFilename(item) || "";
    const parts = [filename, App.statusLabel(item?.status)];
    if (item?.status_detail) parts.push(item.status_detail);
    if (item?.keypoint_status || item?.shenton_status) {
      parts.push(`关键点 ${App.subStatusLabel(item.keypoint_status)}；Shenton ${App.subStatusLabel(item.shenton_status)}`);
    }
    return parts.filter(Boolean).join(" · ");
  },

  statusLabel: (status) => {
    if (status === "auto") return "自动初标待复核";
    if (status === "in_progress") return "标注未完成";
    if (status === "keypoint_complete") return "关键点完成";
    if (status === "shenton_complete") return "Shenton完成";
    if (status === "done") return "完成";
    if (status === "queued") return "排队中";
    return "未标注";
  },

  statusShortLabel: (status) => {
    if (status === "auto") return "待复核";
    if (status === "in_progress") return "未完成";
    if (status === "keypoint_complete") return "点完成";
    if (status === "shenton_complete") return "线完成";
    if (status === "done") return "完成";
    return "未标注";
  },

  subStatusLabel: (status) => {
    if (status === "complete") return "完成";
    if (status === "auto") return "待复核";
    if (status === "in_progress") return "进行中";
    return "未完成";
  },

  filterLabel: (filter) => {
    if (filter === "all") return "全部";
    if (filter === "auto") return "待复核";
    if (filter === "in_progress") return "未完成";
    if (filter === "keypoint_complete") return "关键点完成";
    if (filter === "shenton_complete") return "Shenton完成";
    if (filter === "done") return "已完成";
    return "未标注";
  },

  loadByName: async (filename, options = {}) => {
    if (App.state.annotation && App.state.currentFilename && App.state.currentFilename !== filename) {
      const saved = await App.saveAnnotation({ silent: true, skipManifest: true });
      if (saved === false) return;
    }
    App.setStatus("正在打开缩略图...");
    try {
      const res = await fetch(`/api/annotation/load-by-name?filename=${encodeURIComponent(filename)}`);
      App.applyAnnotation(await App.readJsonResponse(res, "load failed"));
      if (options.refreshManifest) await App.loadManifest();
      App.scrollActiveThumbnailIntoView();
    } catch (err) {
      App.setStatus(`打开失败: ${err.message}`);
    }
  },

  imageFilename: (item) => item?.image_path?.split("/").pop(),

  currentManifestIndex: () =>
    App.state.manifestImages.findIndex((item) => App.imageFilename(item) === App.state.currentFilename),

  loadAdjacentImage: async (delta) => {
    if (App.state.isNavigating) return;
    const images = App.state.manifestImages || [];
    if (!images.length) {
      App.setStatus("当前工作区没有图片");
      return;
    }
    let index = App.currentManifestIndex();
    if (index < 0) {
      index = delta > 0 ? -1 : images.length;
    }
    const nextIndex = index + delta;
    if (nextIndex < 0) {
      App.setStatus("已经是第一张");
      return;
    }
    if (nextIndex >= images.length) {
      App.setStatus("已经是最后一张");
      return;
    }
    const filename = App.imageFilename(images[nextIndex]);
    if (!filename) return;
    App.state.isNavigating = true;
    try {
      await App.loadByName(filename, { skipManifest: true });
    } finally {
      App.state.isNavigating = false;
    }
  },

  deleteCurrentImage: async () => {
    const filename = App.state.currentFilename;
    if (!filename) {
      App.setStatus("请先打开一张图片");
      return;
    }
    const confirmed = window.confirm(
      `确认从当前列表移除这张图片吗？\n\n${filename}\n\n图片、同名标注 JSON 和 YOLO txt 会移动到图片同目录的 trash 文件夹，缓存预览会清理。`,
    );
    if (!confirmed) return;

    const imagesBeforeDelete = App.state.manifestImages || [];
    const currentIndex = App.currentManifestIndex();
    App.setStatus("正在移动当前图片到 trash...");
    try {
      const res = await fetch(`/api/annotation/image/${encodeURIComponent(filename)}`, { method: "DELETE" });
      await App.readJsonResponse(res, "delete image failed");
      await App.loadManifest();
      const remaining = App.state.manifestImages || imagesBeforeDelete.filter((item) => App.imageFilename(item) !== filename);
      const nextItem = remaining[Math.min(Math.max(currentIndex, 0), Math.max(remaining.length - 1, 0))];
      App.state.annotation = null;
      App.state.currentFilename = null;
      App.state.image = null;
      App.state.imageBaseUrl = "";
      App.state.lastSave = null;
      App.state.lastSavedSnapshot = "";
      if (nextItem) {
        await App.loadByName(App.imageFilename(nextItem), { skipManifest: true });
        App.setStatus("已移动当前图片到 trash，并打开下一张");
      } else {
        App.clearWorkspaceView("已移动当前图片到 trash，当前文件夹没有更多图片");
        await App.loadManifest();
      }
    } catch (err) {
      App.setStatus(`删除失败: ${err.message}`);
    }
  },

  keypointsConfirmed: () => {
    const review = App.state.annotation?.review || {};
    return review.manual_keypoints_complete?.status === "confirmed";
  },

  visibleKeypointCount: () => {
    if (!App.state.annotation) return 0;
    return Object.values(App.state.annotation.keypoints || {}).filter((point) => App.pointIsVisible(point)).length;
  },

  confirmKeypointsComplete: async () => {
    if (!App.state.annotation) {
      App.setStatus("请先打开一张图片");
      return;
    }
    const visible = App.visibleKeypointCount();
    const missing = Math.max(0, 22 - visible);
    const message =
      missing > 0
        ? `当前还有 ${missing} 个关键点缺失。\n\n仍要人工确认本图关键点已完成吗？`
        : "确认本图关键点已经人工复核完成吗？";
    if (!window.confirm(message)) return;
    App.pushHistory();
    App.state.annotation.review = App.state.annotation.review || {};
    App.state.annotation.review.manual_keypoints_complete = {
      status: "confirmed",
      visible_count: visible,
      missing_count: missing,
      updated_at: new Date().toISOString(),
      annotator: document.getElementById("annotatorInput")?.value?.trim() || "default",
    };
    App.refreshAll();
    const saved = await App.saveAnnotation();
    App.setStatus(saved ? "已确认关键点完成" : "关键点确认保存失败");
  },

  shentonConfirmed: () => {
    const review = App.state.annotation?.review || {};
    return review.manual_shenton_complete?.status === "confirmed";
  },

  confirmShentonComplete: async () => {
    if (!App.state.annotation) {
      App.setStatus("请先打开一张图片");
      return;
    }
    const shenton = App.state.annotation.shenton_curves || {};
    const review = App.state.annotation.shenton_review || {};
    const sides = ["left", "right"];
    let completeSides = 0;
    for (const side of sides) {
      const curves = shenton[side] || {};
      let curvesOk = true;
      for (const seg of ["obturator_upper_curve", "femoral_neck_inner_lower_curve"]) {
        if ((curves[seg]?.points || []).length < 3) {
          curvesOk = false;
          break;
        }
      }
      const sideReview = review[side]?.status || "not_reviewed";
      const sideReviewed = ["continuous", "discontinuous", "uncertain"].includes(sideReview);
      if (curvesOk && sideReviewed) completeSides += 1;
    }
    const missing = Math.max(0, 2 - completeSides);
    const message =
      missing > 0
        ? `当前还有 ${missing} 侧沈通线未完成。\n\n仍要人工确认本图沈通线已完成吗？`
        : "确认本图沈通线已经人工复核完成吗？";
    if (!window.confirm(message)) return;
    App.pushHistory();
    App.state.annotation.review = App.state.annotation.review || {};
    App.state.annotation.review.manual_shenton_complete = {
      status: "confirmed",
      complete_sides: completeSides,
      updated_at: new Date().toISOString(),
      annotator: document.getElementById("annotatorInput")?.value?.trim() || "default",
    };
    App.refreshAll();
    console.log("confirmShentonComplete: saving annotation with review", JSON.stringify(App.state.annotation.review));
    const saved = await App.saveAnnotation();
    console.log("confirmShentonComplete: save returned", saved, "manifest entry:", App.state.manifestByFilename.get(App.state.annotation?.image?.filename));
    App.setStatus(saved ? "已确认沈通线完成" : "沈通线确认保存失败");
  },

  scrollActiveThumbnailIntoView: () => {
    const active = document.querySelector(".thumb.active");
    if (!active) return;
    active.scrollIntoView({ block: "nearest", inline: "center" });
  },

  clearWorkspaceView: (message = "未打开图像") => {
    App.state.image = null;
    App.state.imageBaseUrl = "";
    App.state.imageView = "enhanced";
    App.state.annotation = null;
    App.state.currentFilename = null;
    App.state.selectedPoint = null;
    App.state.selectedShentonPoint = null;
    App.state.selectedConnection = null;
    App.state.pendingConnectionStart = null;
    App.state.roiStart = null;
    App.state.roiDraft = null;
    App.state.manifestImages = [];
    App.state.progress = null;
    App.state.importReport = null;
    App.state.lastSave = null;
    App.state.lastSavedSnapshot = "";
    App.state.manifestByFilename = new Map();
    App.state.thumbByFilename = new Map();
    App.state.history = [];
    App.state.historyIndex = -1;
    App.state.transform = { x: 0, y: 0, scale: 1 };
    document.getElementById("imageTitle").textContent = "未打开图像";
    document.getElementById("warningList").innerHTML = "";
    document.getElementById("connectionList").innerHTML = "";
    document.getElementById("thumbStrip").innerHTML = "";
    App.renderProgressSummary();
    App.renderShentonPanel();
    App.renderMeasurements();
    App.syncDisplayToggles();
    document.getElementById("selectedConnectionLabel").textContent = "未选中连线";
    document.querySelectorAll(".point-row").forEach((row) => {
      row.classList.add("missing");
      row.classList.remove("selected");
      const checkbox = row.querySelector("input");
      const badge = row.querySelector(".source-badge");
      if (checkbox) checkbox.checked = false;
      if (badge) {
        badge.className = "source-badge missing";
        badge.textContent = "缺失";
      }
    });
    App.updateSummary();
    App.updateSelectedBox();
    App.setStatus(message);
  },

  startAutoDetectPolling: () => {
    clearInterval(App.state.autoDetectPollTimer);
    App.pollAutoDetectStatus();
    App.state.autoDetectPollTimer = setInterval(App.pollAutoDetectStatus, 1600);
  },

  pollAutoDetectStatus: async () => {
    try {
      const res = await fetch("/api/annotation/auto-detect/status");
      if (!res.ok) throw new Error(await res.text());
      const next = await App.readJsonResponse(res, "auto detect status failed");
      const previous = JSON.stringify(App.state.autoDetectStatus || {});
      App.state.autoDetectStatus = next;
      App.renderAutoDetectStatus();
      const changed = previous !== JSON.stringify(next || {});
      if (changed) {
        await App.loadManifest();
        App.maybeRefreshActiveFromBackground();
      }
      if (!next.running && !next.pending && !next.processing) {
        clearInterval(App.state.autoDetectPollTimer);
        App.state.autoDetectPollTimer = null;
      }
    } catch (err) {
      console.warn(err);
    }
  },

  renderAutoDetectStatus: () => {
    const target = document.getElementById("autoDetectQueueText");
    if (!target) return;
    const status = App.state.autoDetectStatus;
    if (!status || !status.total) {
      target.textContent = "未排队";
      return;
    }
    const finished = (status.done || 0) + (status.skipped || 0) + (status.failed || 0);
    const processing = status.processing ? `，正在处理 ${status.processing}` : "";
    target.textContent = `${finished}/${status.total} 完成，待处理 ${status.pending || 0}${processing}`;
  },

  maybeRefreshActiveFromBackground: () => {
    if (!App.state.currentFilename || !App.state.annotation) return;
    const source = App.state.annotation.auto_initialization?.source;
    if (source !== "queued") return;
    const manualVisible = Object.values(App.state.annotation.keypoints || {}).some(
      (point) => App.pointIsVisible(point) && point.source === "manual",
    );
    if (manualVisible) return;
    const item = App.state.manifestImages.find((entry) => entry.image_path.split("/").pop() === App.state.currentFilename);
    if (!item || item.status === "pending") return;
    App.loadByName(App.state.currentFilename, { skipManifest: true });
  },

  DEFAULT_CONNECTION_PAIRS: [
    ["left_femoral_shaft_prox", "left_femoral_shaft_dist"],
    ["left_femoral_head_medial", "left_femoral_head_center"],
    ["left_femoral_head_center", "left_femoral_head_lateral"],
    ["left_femoral_head_center", "left_femoral_neck_axis_center"],
    ["left_acetabular_outer", "left_triradiate_center"],
    ["right_femoral_shaft_prox", "right_femoral_shaft_dist"],
    ["right_femoral_head_medial", "right_femoral_head_center"],
    ["right_femoral_head_center", "right_femoral_head_lateral"],
    ["right_femoral_head_center", "right_femoral_neck_axis_center"],
    ["right_acetabular_outer", "right_triradiate_center"],
    ["left_triradiate_center", "right_triradiate_center"],
  ],

  ensureDefaultConnections: (annotation) => {
    const manual = (annotation.connections || []).filter((c) => c.source !== "default");
    const existingDefaultPairs = new Set(
      (annotation.connections || []).filter((c) => c.source === "default").map((c) => `${c.point_a}__${c.point_b}`)
    );
    const defaults = App.DEFAULT_CONNECTION_PAIRS.map(([a, b]) => {
      const old = (annotation.connections || []).find(
        (c) => c.source === "default" && c.point_a === a && c.point_b === b
      );
      return {
        id: old?.id || `default_${a}__${b}`,
        point_a: a,
        point_b: b,
        label: "默认连线",
        source: "default",
        visible: true,
        updated_at: old?.updated_at || new Date().toISOString(),
        annotator: "",
      };
    });
    annotation.connections = [...manual, ...defaults];
  },

  applyAnnotation: (annotation) => {
    annotation.connections = Array.isArray(annotation.connections) ? annotation.connections : [];
    App.ensureDefaultConnections(annotation);
    App.normalizeShenton(annotation);
    App.normalizeRoiCrop(annotation);
    App.normalizeScanTransform(annotation);
    App.state.annotation = annotation;
    App.state.currentFilename = annotation.image.filename;
    App.state.lastSavedSnapshot = App.annotationSnapshot(annotation);
    App.state.selectedPoint = null;
    App.state.selectedShentonPoint = null;
    App.state.selectedScanCorner = null;
    App.state.selectedConnection = null;
    App.state.pendingConnectionStart = null;
    App.state.activePointKey = "left_acetabular_outer";
    App.state.lastSave = null;
    App.state.imageView = "enhanced";
    App.state.roiStart = null;
    App.state.roiDraft = null;
    App.state.imageBaseUrl = annotation.image_url || `/api/annotation/image/${encodeURIComponent(annotation.image.filename)}`;
    App.setActiveThumbnail(annotation.image.filename);
    document.getElementById("annotatorInput").value = annotation.annotator?.user_id || App.state.settings.annotator || "default";
    document.getElementById("imageTitle").textContent = annotation.image.filename;
    App.loadImage(App.imageUrlForCurrentView());
    App.resetHistory();
    App.refreshAll();
    App.renderWarnings();
    App.renderSaveInfo();
    App.syncImageViewButtons();
    App.scheduleMeasurementCompute();
  },

  loadImage: (src) => {
    const image = new Image();
    image.onload = () => {
      App.state.image = image;
      App.fitToScreen();
      App.resetHistory();
      App.setStatus("已载入");
    };
    image.onerror = () => {
      App.setStatus("图片载入失败，请检查格式或尝试重新导入");
    };
    image.src = src;
  },

  imageUrlForCurrentView: () => {
    const base = App.state.imageBaseUrl || (App.state.currentFilename ? `/api/annotation/image/${encodeURIComponent(App.state.currentFilename)}` : "");
    if (!base) return "";
    const separator = base.includes("?") ? "&" : "?";
    return App.state.imageView === "enhanced" ? `${base}${separator}enhanced=true` : base;
  },

  setImageView: (view) => {
    App.state.imageView = view === "enhanced" ? "enhanced" : "original";
    App.syncImageViewButtons();
    if (App.state.imageBaseUrl) App.loadImage(App.imageUrlForCurrentView());
  },

  syncImageViewButtons: () => {
    document.querySelectorAll("[data-image-view]").forEach((button) => {
      button.classList.toggle("active", button.dataset.imageView === App.state.imageView);
    });
  },

  normalizeRoiCrop: (annotation) => {
    const current = annotation.roi_crop && typeof annotation.roi_crop === "object" ? annotation.roi_crop : {};
    const enabled = Boolean(current.enabled);
    const x = Number(current.x);
    const y = Number(current.y);
    const width = Number(current.width);
    const height = Number(current.height);
    annotation.roi_crop = {
      enabled: enabled && Number.isFinite(x) && Number.isFinite(y) && width >= 8 && height >= 8,
      x: Number.isFinite(x) ? x : null,
      y: Number.isFinite(y) ? y : null,
      width: Number.isFinite(width) ? width : null,
      height: Number.isFinite(height) ? height : null,
      source: current.source || "manual",
      updated_at: current.updated_at || new Date().toISOString(),
      annotator: current.annotator || document.getElementById("annotatorInput")?.value?.trim() || "default",
    };
    if (!annotation.roi_crop.enabled) {
      annotation.roi_crop.x = null;
      annotation.roi_crop.y = null;
      annotation.roi_crop.width = null;
      annotation.roi_crop.height = null;
    }
  },

  normalizeScanTransform: (annotation) => {
    const current = annotation.scan_transform && typeof annotation.scan_transform === "object" ? annotation.scan_transform : {};
    const rawCorners = Array.isArray(current.corners) ? current.corners : [];
    const corners = rawCorners
      .slice(0, 4)
      .map((point) => ({
        x: Number(point?.x),
        y: Number(point?.y),
      }))
      .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
      .map((point) => ({
        x: App.clamp(point.x, 0, annotation.image.width || 1),
        y: App.clamp(point.y, 0, annotation.image.height || 1),
      }));
    const ordered = corners.length === 4 ? App.orderScanCorners(corners) : corners;
    annotation.scan_transform = {
      enabled: Boolean(current.enabled) && ordered.length === 4 && App.scanPolygonArea(ordered) >= 64,
      corners: ordered,
      mode: current.mode || "manual_four_corners",
      source: current.source || "manual",
      updated_at: current.updated_at || new Date().toISOString(),
      annotator: current.annotator || document.getElementById("annotatorInput")?.value?.trim() || "default",
    };
  },

  normalizeShenton: (annotation) => {
    annotation.shenton_curves = annotation.shenton_curves && typeof annotation.shenton_curves === "object" ? annotation.shenton_curves : {};
    annotation.shenton_review = annotation.shenton_review && typeof annotation.shenton_review === "object" ? annotation.shenton_review : {};
    annotation.shenton_adjustments =
      annotation.shenton_adjustments && typeof annotation.shenton_adjustments === "object" ? annotation.shenton_adjustments : {};
    ["left", "right"].forEach((side) => {
      annotation.shenton_curves[side] = annotation.shenton_curves[side] || {};
      ["obturator_upper_curve", "femoral_neck_inner_lower_curve"].forEach((segment) => {
        const current = annotation.shenton_curves[side][segment] || {};
        annotation.shenton_curves[side][segment] = {
          type: current.type || "polyline",
          points: Array.isArray(current.points) ? current.points : [],
          source: current.source || "manual",
          updated_at: current.updated_at || new Date().toISOString(),
          annotator: current.annotator || document.getElementById("annotatorInput")?.value?.trim() || "default",
        };
      });
      const review = annotation.shenton_review[side] || {};
      const allowed = ["continuous", "discontinuous", "uncertain", "not_reviewed"];
      annotation.shenton_review[side] = {
        status: allowed.includes(review.status) ? review.status : "not_reviewed",
        updated_at: review.updated_at || new Date().toISOString(),
        annotator: review.annotator || document.getElementById("annotatorInput")?.value?.trim() || "default",
      };
      const adjustments = annotation.shenton_adjustments[side] || {};
      const rawIntersection = adjustments.extension_intersection || {};
      const x = Number(rawIntersection.x);
      const y = Number(rawIntersection.y);
      const enabled = Boolean(rawIntersection.enabled) && Number.isFinite(x) && Number.isFinite(y);
      annotation.shenton_adjustments[side] = {
        extension_intersection: {
          enabled,
          x: enabled ? App.clamp(x, 0, annotation.image.width || 1) : null,
          y: enabled ? App.clamp(y, 0, annotation.image.height || 1) : null,
          source: rawIntersection.source || "manual",
          updated_at: rawIntersection.updated_at || new Date().toISOString(),
          annotator: rawIntersection.annotator || document.getElementById("annotatorInput")?.value?.trim() || "default",
        },
      };
    });
  },

  currentShentonSegment: () => {
    if (!App.state.annotation) return null;
    App.normalizeShenton(App.state.annotation);
    return App.state.annotation.shenton_curves[App.state.shentonSide][App.state.shentonSegment];
  },

  hasShentonPoints: () => {
    const curves = App.state.annotation?.shenton_curves || {};
    return ["left", "right"].some((side) =>
      ["obturator_upper_curve", "femoral_neck_inner_lower_curve"].some(
        (segment) => (curves[side]?.[segment]?.points || []).length > 0,
      ),
    );
  },

  setShentonSide: (side) => {
    App.state.shentonSide = side === "right" ? "right" : "left";
    App.state.selectedShentonPoint = null;
    App.renderShentonPanel();
  },

  setShentonSegment: (segment) => {
    App.state.shentonSegment =
      segment === "femoral_neck_inner_lower_curve" ? "femoral_neck_inner_lower_curve" : "obturator_upper_curve";
    App.state.selectedShentonPoint = null;
    App.renderShentonPanel();
  },

  addShentonPoint: (imagePos) => {
    const segment = App.currentShentonSegment();
    if (!segment || !App.state.image) return;
    App.pushHistory();
    segment.points.push({
      x: App.clamp(imagePos.x, 0, App.state.image.width),
      y: App.clamp(imagePos.y, 0, App.state.image.height),
    });
    segment.updated_at = new Date().toISOString();
    segment.annotator = document.getElementById("annotatorInput").value.trim() || "default";
    App.state.selectedPoint = null;
    App.state.selectedConnection = null;
    App.state.selectedShentonPoint = {
      side: App.state.shentonSide,
      segment: App.state.shentonSegment,
      index: segment.points.length - 1,
    };
    App.refreshAll();
    App.pushHistory();
    App.scheduleMeasurementCompute();
    App.scheduleAutosave();
  },

  undoShentonPoint: () => {
    const segment = App.currentShentonSegment();
    if (!segment || !segment.points.length) return;
    App.pushHistory();
    segment.points.pop();
    segment.updated_at = new Date().toISOString();
    App.state.selectedShentonPoint = null;
    App.refreshAll();
    App.pushHistory();
    App.scheduleMeasurementCompute();
    App.scheduleAutosave();
  },

  clearShentonSegment: () => {
    const segment = App.currentShentonSegment();
    if (!segment || !segment.points.length) return;
    App.pushHistory();
    segment.points = [];
    segment.updated_at = new Date().toISOString();
    App.state.selectedShentonPoint = null;
    App.refreshAll();
    App.pushHistory();
    App.scheduleMeasurementCompute();
    App.scheduleAutosave();
  },

  deleteSelectedShentonPoint: () => {
    const selected = App.state.selectedShentonPoint;
    if (!selected || !App.state.annotation) return;
    const segment = App.state.annotation.shenton_curves?.[selected.side]?.[selected.segment];
    if (!segment?.points?.[selected.index]) return;
    App.pushHistory();
    segment.points.splice(selected.index, 1);
    segment.updated_at = new Date().toISOString();
    App.state.selectedShentonPoint = null;
    App.refreshAll();
    App.pushHistory();
    App.scheduleMeasurementCompute();
    App.scheduleAutosave();
  },

  updateShentonReview: () => {
    if (!App.state.annotation) return;
    App.normalizeShenton(App.state.annotation);
    App.pushHistory();
    const review = App.state.annotation.shenton_review[App.state.shentonSide];
    review.status = document.getElementById("shentonReviewSelect").value || "not_reviewed";
    review.updated_at = new Date().toISOString();
    review.annotator = document.getElementById("annotatorInput").value.trim() || "default";
    App.refreshAll();
    App.pushHistory();
    App.scheduleMeasurementCompute();
    App.scheduleAutosave();
  },

  renderShentonPanel: () => {
    document.querySelectorAll("[data-shenton-side]").forEach((button) => {
      button.classList.toggle("active", button.dataset.shentonSide === App.state.shentonSide);
    });
    document.querySelectorAll("[data-shenton-segment]").forEach((button) => {
      button.classList.toggle("active", button.dataset.shentonSegment === App.state.shentonSegment);
    });
    const select = document.getElementById("shentonReviewSelect");
    const hint = document.getElementById("shentonHintText");
    if (!select || !hint) return;
    if (!App.state.annotation) {
      select.value = "not_reviewed";
      hint.textContent = "每侧两段曲线，每段至少 3 个点，点越多越贴合曲线；本功能仅用于研究复核辅助。";
      return;
    }
    App.normalizeShenton(App.state.annotation);
    select.value = App.state.annotation.shenton_review[App.state.shentonSide]?.status || "not_reviewed";
    const segment = App.currentShentonSegment();
    const count = segment?.points?.length || 0;
    hint.textContent = `当前段 ${count} 个点；至少 3 点后可计算 Shenton 连续性候选，点越多越贴合曲线。`;
  },

  scheduleMeasurementCompute: () => {
    clearTimeout(App.state.measurementTimer);
    App.state.measurementTimer = setTimeout(App.computeMeasurements, 250);
  },

  computeMeasurements: async () => {
    if (!App.state.annotation) return;
    App.normalizeShenton(App.state.annotation);
    try {
      const res = await fetch("/api/annotation/measurements/compute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(App.state.annotation),
      });
      const measurements = await App.readJsonResponse(res, "measurements failed");
      App.state.annotation.measurements_snapshot = {
        ...(App.state.annotation.measurements_snapshot || {}),
        ...measurements,
      };
      App.renderMeasurements();
    } catch (err) {
      console.warn("measurement compute failed", err);
    }
  },

  renderMeasurements: () => {
    const summary = document.getElementById("shentonSummaryText");
    const clinical = document.getElementById("clinicalParameterText");
    const acetabular = document.getElementById("acetabularDepthText");
    if (!summary || !clinical || !acetabular) return;
    if (!App.state.showMeasurements) {
      summary.textContent = "测量显示已关闭";
      clinical.textContent = "关键指标：已隐藏";
      acetabular.textContent = "髋臼深度：已隐藏";
      return;
    }
    const snapshot = App.state.annotation?.measurements_snapshot || {};
    const shenton = snapshot.shenton || {};
    const sideText = (side) => {
      const item = shenton[side] || {};
      if (item.status === "unavailable") return `${side === "left" ? "左" : "右"}：曲线不足`;
      const gap = typeof item.gap_px === "number" ? `${item.gap_px.toFixed(1)}px` : "-";
      const gapMm = typeof item.gap_mm === "number" ? ` / ${item.gap_mm.toFixed(2)}mm` : "";
      return `${side === "left" ? "左" : "右"}：端点 gap ${gap}${gapMm}`;
    };
    summary.textContent = `Shenton：${sideText("left")}；${sideText("right")}`;
    const clinicalParameters = snapshot.clinical_parameters || {};
    const formatNumber = (value, suffix = "°") => (typeof value === "number" ? `${value.toFixed(1)}${suffix}` : "-");
    const parameterText = (side) => {
      const item = clinicalParameters[side] || {};
      const label = side === "left" ? "左" : "右";
      if (item.status !== "computed") return `${label}：关键点不足`;
      return `${label}：AI ${formatNumber(item.ai_tonnis_angle_deg)}，Sharp ${formatNumber(item.sharp_angle_deg)}，CE ${formatNumber(item.ce_angle_deg)}，颈干 ${formatNumber(item.neck_shaft_angle_deg)}`;
    };
    clinical.textContent = `关键指标：${parameterText("left")}；${parameterText("right")}`;
    const depth = snapshot.acetabular_depth || {};
    const depthText = (side) => {
      const item = depth[side] || {};
      const label = side === "left" ? "左" : "右";
      if (item.status !== "computed") return `${label}：关键点不足`;
      const px = typeof item.value_px === "number" ? `${item.value_px.toFixed(1)}px` : "-";
      const mm = typeof item.value_mm === "number" ? ` / ${item.value_mm.toFixed(2)}mm` : "";
      return `${label}：${px}${mm}`;
    };
    acetabular.textContent = `髋臼深度：${depthText("left")}；${depthText("right")}`;
  },

  syncDisplayToggles: () => {
    const toggles = {
      showDefaultConnectionsToggle: App.state.showDefaultConnections,
      showManualConnectionsToggle: App.state.showManualConnections,
      showShentonToggle: App.state.showShenton,
      showMeasurementsToggle: App.state.showMeasurements,
      showPointLabelsToggle: App.state.showLabels,
      showPoint10And11Toggle: App.state.showPoint10And11,
    };
    Object.entries(toggles).forEach(([id, value]) => {
      const element = document.getElementById(id);
      if (element) element.checked = Boolean(value);
    });
  },

  annotationPayloadForSave: (annotation = App.state.annotation) => {
    if (!annotation) return null;
    const payload = JSON.parse(JSON.stringify(annotation));
    delete payload.image_url;
    return payload;
  },

  annotationSnapshot: (annotation = App.state.annotation) => {
    const payload = App.annotationPayloadForSave(annotation);
    return payload ? JSON.stringify(payload) : "";
  },

  saveAnnotation: async (options = {}) => {
    if (!App.state.annotation) return true;
    App.state.annotation.annotator = App.state.annotation.annotator || {};
    App.state.annotation.annotator.user_id = document.getElementById("annotatorInput").value.trim() || "default";
    App.state.settings.annotator = App.state.annotation.annotator.user_id;
    if (App.state.image) {
      App.state.annotation.image.width = App.state.image.naturalWidth || App.state.image.width;
      App.state.annotation.image.height = App.state.image.naturalHeight || App.state.image.height;
    }
    App.normalizeRoiCrop(App.state.annotation);
    App.normalizeScanTransform(App.state.annotation);
    const snapshot = App.annotationSnapshot();
    if (options.silent && snapshot === App.state.lastSavedSnapshot) {
      return true;
    }
    if (!options.silent) App.setStatus("正在保存...");
    try {
      const saveUrl = `/api/annotation/save${options.skipManifest ? "?skip_manifest=true" : ""}`;
      const res = await fetch(saveUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(App.annotationPayloadForSave()),
      });
      const payload = await App.readJsonResponse(res, "save failed");
      console.log("saveAnnotation: server response", { status: payload.annotation_status, progress: payload.annotation_progress });
      if (payload.measurements_snapshot) {
        App.state.annotation.measurements_snapshot = payload.measurements_snapshot;
        App.renderMeasurements();
      }
      App.state.lastSave = {
        filename: App.state.annotation.image.filename,
        timeText: new Date().toLocaleTimeString(),
        annotationPath: payload.annotation_path || "annotations",
        labelPath: payload.label_path || "同名 txt",
      };
      App.state.lastSavedSnapshot = App.annotationSnapshot();
      App.updateManifestEntryStatus(
        App.state.annotation.image.filename,
        payload.annotation_progress || { status: payload.annotation_status },
      );
      App.state.progress = payload.progress || null;
      App.setStatus(options.silent ? "已自动保存" : "已保存");
      if (!options.skipManifest && !payload.progress) {
        await App.loadManifest();
      } else {
        App.renderProgressSummary();
      }
      App.renderSaveInfo();
      return true;
    } catch (err) {
      App.setStatus(`保存失败: ${err.message}`);
      return false;
    }
  },

  scheduleAutosave: () => {
    if (!App.state.settings.autosave || !App.state.annotation) return;
    clearTimeout(App.state.autosaveTimer);
    App.state.autosaveTimer = setTimeout(() => App.saveAnnotation({ silent: true }), 900);
  },

  resetHistory: () => {
    if (!App.state.annotation) {
      App.state.history = [];
      App.state.historyIndex = -1;
      return;
    }
    App.state.history = [JSON.stringify(App.state.annotation)];
    App.state.historyIndex = 0;
  },

  pushHistory: () => {
    if (!App.state.annotation) return;
    const snapshot = JSON.stringify(App.state.annotation);
    if (App.state.history[App.state.historyIndex] === snapshot) return;
    App.state.history = App.state.history.slice(0, App.state.historyIndex + 1);
    App.state.history.push(snapshot);
    App.state.historyIndex = App.state.history.length - 1;
  },

  restoreHistory: () => {
    App.state.annotation = JSON.parse(App.state.history[App.state.historyIndex]);
    App.refreshAll();
  },

  undo: () => {
    if (App.state.historyIndex <= 0) return;
    App.state.historyIndex -= 1;
    App.restoreHistory();
    App.scheduleAutosave();
  },

  redo: () => {
    if (App.state.historyIndex >= App.state.history.length - 1) return;
    App.state.historyIndex += 1;
    App.restoreHistory();
    App.scheduleAutosave();
  },

  refreshAll: () => {
    App.refreshPointList();
    App.renderConnectionList();
    App.renderShentonPanel();
    App.renderMeasurements();
    App.renderRoiPanel();
    App.renderScanPanel();
    App.syncDisplayToggles();
    App.updateSummary();
    App.renderKeypointConfirmation();
    App.renderShentonConfirmation();
    App.updateSelectedBox();
  },

  selectPoint: (key) => {
    App.state.selectedPoint = key;
    App.state.selectedShentonPoint = null;
    App.state.selectedScanCorner = null;
    App.state.selectedConnection = null;
    document.querySelectorAll(".point-row").forEach((row) => row.classList.toggle("selected", row.dataset.key === key));
    App.renderConnectionList();
    App.updateSelectedBox();
  },

  selectConnection: (id) => {
    App.state.selectedConnection = id;
    App.state.selectedPoint = null;
    App.state.selectedShentonPoint = null;
    App.state.selectedScanCorner = null;
    document.querySelectorAll(".point-row").forEach((row) => row.classList.remove("selected"));
    App.renderConnectionList();
    App.updateSelectedBox();
  },

  pointIsVisible: (point) => Boolean(point && point.visible && point.visibility > 0 && point.x !== null && point.y !== null),

  pointIsDefaultHidden: (point) => DEFAULT_HIDDEN_POINT_NUMBERS.has(Number(point?.number)),

  pointIsDisplayed: (key, point) =>
    App.pointIsVisible(point) &&
    (App.state.showPoint10And11 ||
      !App.pointIsDefaultHidden(point) ||
      key === App.state.selectedPoint ||
      key === App.state.pendingConnectionStart),

  togglePointVisibility: (key, visible) => {
    if (!App.state.annotation?.keypoints?.[key]) return;
    App.pushHistory();
    const point = App.state.annotation.keypoints[key];
    if (visible && (point.x === null || point.y === null)) {
      const center = App.imageCenter();
      point.x = center.x;
      point.y = center.y;
    }
    point.visible = visible;
    point.visibility = visible ? 2 : 0;
    point.source = "manual";
    point.confidence = visible ? 1 : 0;
    point.updated_at = new Date().toISOString();
    App.refreshAll();
    App.pushHistory();
    App.scheduleMeasurementCompute();
    App.scheduleAutosave();
  },

  imageCenter: () => {
    if (!App.state.image) return { x: 0, y: 0 };
    return { x: App.state.image.width / 2, y: App.state.image.height / 2 };
  },

  placePoint: (key, imagePos) => {
    if (!App.state.annotation?.keypoints?.[key]) return;
    App.pushHistory();
    const point = App.state.annotation.keypoints[key];
    point.x = App.clamp(imagePos.x, 0, App.state.image.width);
    point.y = App.clamp(imagePos.y, 0, App.state.image.height);
    point.visible = true;
    point.visibility = 2;
    point.source = "manual";
    point.confidence = 1;
    point.updated_at = new Date().toISOString();
    point.annotator = document.getElementById("annotatorInput").value.trim() || "default";
    App.selectPoint(key);
    App.refreshAll();
    App.pushHistory();
    App.scheduleMeasurementCompute();
    App.scheduleAutosave();
  },

  markMissing: (key) => {
    if (!App.state.annotation?.keypoints?.[key]) return;
    App.pushHistory();
    const point = App.state.annotation.keypoints[key];
    point.x = null;
    point.y = null;
    point.visible = false;
    point.visibility = 0;
    point.source = "manual";
    point.confidence = 0;
    point.updated_at = new Date().toISOString();
    App.refreshAll();
    App.pushHistory();
    App.scheduleAutosave();
  },

  clearCurrentImagePoints: () => {
    if (!App.state.annotation?.keypoints) return;
    const visibleCount = Object.values(App.state.annotation.keypoints).filter(App.pointIsVisible).length;
    if (!visibleCount) {
      App.setStatus("当前图片没有可清空的关键点");
      return;
    }
    if (!window.confirm(`确定清空当前图片的 ${visibleCount} 个已标关键点吗？`)) return;

    App.pushHistory();
    const annotator = document.getElementById("annotatorInput").value.trim() || "default";
    Object.values(App.state.annotation.keypoints).forEach((point) => {
      point.x = null;
      point.y = null;
      point.visible = false;
      point.visibility = 0;
      point.source = "manual";
      point.confidence = 0;
      point.updated_at = new Date().toISOString();
      point.annotator = annotator;
    });
    App.state.selectedPoint = null;
    App.state.pendingConnectionStart = null;
    App.refreshAll();
    App.pushHistory();
    App.scheduleMeasurementCompute();
    App.scheduleAutosave();
    App.setStatus("已清空当前图片的全部关键点");
  },

  addConnection: (pointA, pointB) => {
    if (!pointA || !pointB || pointA === pointB) return;
    App.pushHistory();
    const existing = App.state.annotation.connections.find(
      (item) =>
        (item.point_a === pointA && item.point_b === pointB) ||
        (item.point_a === pointB && item.point_b === pointA),
    );
    if (existing) {
      existing.visible = true;
      existing.updated_at = new Date().toISOString();
      App.selectConnection(existing.id);
    } else {
      const id = `manual_${pointA}__${pointB}_${Date.now()}`;
      App.state.annotation.connections.push({
        id,
        point_a: pointA,
        point_b: pointB,
        label: "人工连线",
        source: "manual",
        visible: true,
        updated_at: new Date().toISOString(),
        annotator: document.getElementById("annotatorInput").value.trim() || "default",
      });
      App.selectConnection(id);
    }
    App.state.pendingConnectionStart = null;
    App.refreshAll();
    App.pushHistory();
    App.scheduleAutosave();
  },

  deleteSelectedConnection: () => {
    if (!App.state.annotation || !App.state.selectedConnection) return;
    App.pushHistory();
    const index = App.state.annotation.connections.findIndex((item) => item.id === App.state.selectedConnection);
    if (index < 0) return;
    const connection = App.state.annotation.connections[index];
    if (connection.source === "default") {
      connection.visible = false;
      connection.updated_at = new Date().toISOString();
    } else {
      App.state.annotation.connections.splice(index, 1);
    }
    App.state.selectedConnection = null;
    App.refreshAll();
    App.pushHistory();
    App.scheduleAutosave();
  },

  handleMouseDown: (event) => {
    if (event.button === 2 || !App.state.annotation || !App.state.image) return;
    const pos = App.mousePos(event);
    const imagePos = App.toImageCoords(pos.x, pos.y);
    App.hideContextMenu();
    if (event.button === 1 || App.state.spaceHeld) {
      App.state.isPanning = true;
      App.state.lastMouse = pos;
      return;
    }
    if (App.state.activeTool === "roi") {
      App.state.isDrawingRoi = true;
      App.state.dragStarted = false;
      App.state.roiStart = imagePos;
      App.state.roiDraft = null;
      App.state.lastMouse = pos;
      return;
    }
    if (App.state.activeTool === "scan") {
      const hitScan = App.hitTestScanCorner(pos.x, pos.y);
      if (hitScan !== null) {
        App.state.selectedScanCorner = hitScan;
        App.state.selectedPoint = null;
        App.state.selectedShentonPoint = null;
        App.state.selectedConnection = null;
        App.state.isDraggingScanCorner = true;
        App.state.dragStarted = false;
        App.state.lastMouse = pos;
        App.refreshAll();
        return;
      }
      App.addScanCorner(imagePos);
      return;
    }
    if (App.state.activeTool === "shenton") {
      const hitShenton = App.hitTestShentonPoint(pos.x, pos.y);
      if (hitShenton) {
        App.state.selectedShentonPoint = hitShenton;
        App.state.selectedPoint = null;
        App.state.selectedConnection = null;
        App.state.isDraggingShenton = true;
        App.state.dragStarted = false;
        App.state.lastMouse = pos;
        App.refreshAll();
        return;
      }
      App.addShentonPoint(imagePos);
      return;
    }
    if (App.state.activeTool === "point") {
      App.placePoint(App.state.activePointKey, imagePos);
      return;
    }
    const hitPoint = App.hitTestPoint(pos.x, pos.y);
    if (App.state.activeTool === "line") {
      if (!hitPoint) {
        App.setStatus("连线工具需要点击两个已显示的点");
        return;
      }
      if (!App.state.pendingConnectionStart) {
        App.state.pendingConnectionStart = hitPoint;
        App.selectPoint(hitPoint);
        App.setStatus("请选择第二个点完成连线");
      } else {
        App.addConnection(App.state.pendingConnectionStart, hitPoint);
        App.setStatus("连线已添加");
      }
      return;
    }
    if (hitPoint) {
      App.selectPoint(hitPoint);
      App.state.activePointKey = hitPoint;
      App.state.isDragging = true;
      App.state.dragStarted = false;
      App.state.lastMouse = pos;
      return;
    }
    const hitConnection = App.hitTestConnection(pos.x, pos.y);
    if (hitConnection) {
      App.selectConnection(hitConnection);
      return;
    }
    App.state.isPanning = true;
    App.state.lastMouse = pos;
  },

  handleMouseMove: (event) => {
    const pos = App.mousePos(event);
    if (App.state.isDraggingShenton && App.state.selectedShentonPoint) {
      const selected = App.state.selectedShentonPoint;
      if (!App.state.dragStarted) {
        App.pushHistory();
        App.state.dragStarted = true;
      }
      const imagePos = App.toImageCoords(pos.x, pos.y);
      const segment = App.state.annotation?.shenton_curves?.[selected.side]?.[selected.segment];
      const point = segment?.points?.[selected.index];
      if (!point) return;
      point.x = App.clamp(imagePos.x, 0, App.state.image.width);
      point.y = App.clamp(imagePos.y, 0, App.state.image.height);
      segment.updated_at = new Date().toISOString();
      segment.annotator = document.getElementById("annotatorInput").value.trim() || "default";
      App.renderShentonPanel();
      App.renderMeasurements();
    } else if (App.state.isDraggingScanCorner && App.state.selectedScanCorner !== null) {
      App.normalizeScanTransform(App.state.annotation);
      const scan = App.state.annotation?.scan_transform;
      const point = scan?.corners?.[App.state.selectedScanCorner];
      if (!point) return;
      if (!App.state.dragStarted) {
        App.pushHistory();
        App.state.dragStarted = true;
      }
      const imagePos = App.toImageCoords(pos.x, pos.y);
      point.x = Number(App.clamp(imagePos.x, 0, App.state.image.width).toFixed(2));
      point.y = Number(App.clamp(imagePos.y, 0, App.state.image.height).toFixed(2));
      scan.enabled = scan.corners.length === 4 && App.scanPolygonArea(scan.corners) >= 64;
      scan.updated_at = new Date().toISOString();
      scan.annotator = document.getElementById("annotatorInput").value.trim() || "default";
      App.renderScanPanel();
      App.updateSelectedBox();
    } else if (App.state.isDrawingRoi && App.state.roiStart) {
      const imagePos = App.toImageCoords(pos.x, pos.y);
      App.state.roiDraft = App.normalizedRoiBox(App.state.roiStart, imagePos);
      App.state.dragStarted = Boolean(App.state.roiDraft);
    } else if (App.state.isDragging && App.state.selectedPoint) {
      const point = App.state.annotation.keypoints[App.state.selectedPoint];
      if (!App.state.dragStarted) {
        App.pushHistory();
        App.state.dragStarted = true;
      }
      point.x = App.clamp((point.x || 0) + (pos.x - App.state.lastMouse.x) / App.state.transform.scale, 0, App.state.image.width);
      point.y = App.clamp((point.y || 0) + (pos.y - App.state.lastMouse.y) / App.state.transform.scale, 0, App.state.image.height);
      point.visible = true;
      point.visibility = 2;
      point.source = "manual";
      point.confidence = 1;
      point.updated_at = new Date().toISOString();
      App.state.lastMouse = pos;
      App.refreshPointList();
      App.updateSummary();
    } else if (App.state.isPanning) {
      App.state.transform.x += pos.x - App.state.lastMouse.x;
      App.state.transform.y += pos.y - App.state.lastMouse.y;
      App.state.lastMouse = pos;
    }
  },

  handleMouseUp: () => {
    if (App.state.isDraggingShenton && App.state.dragStarted) {
      App.pushHistory();
      App.scheduleMeasurementCompute();
      App.scheduleAutosave();
    }
    if (App.state.isDraggingScanCorner && App.state.dragStarted) {
      App.pushHistory();
      App.scheduleAutosave();
    }
    if (App.state.isDragging && App.state.dragStarted) {
      App.pushHistory();
      App.scheduleMeasurementCompute();
      App.scheduleAutosave();
    }
    if (App.state.isDrawingRoi && App.state.dragStarted && App.state.roiDraft) {
      App.pushHistory();
      App.setRoiCrop(App.state.roiDraft);
      App.state.roiDraft = null;
      App.pushHistory();
      App.scheduleAutosave();
      App.setStatus("ROI 已更新；识别会优先使用该区域");
    }
    App.state.isDragging = false;
    App.state.isDraggingShenton = false;
    App.state.isDraggingScanCorner = false;
    App.state.isDrawingRoi = false;
    App.state.isPanning = false;
    App.state.dragStarted = false;
    App.state.roiStart = null;
  },

  handleWheel: (event) => {
    if (!App.state.image) return;
    event.preventDefault();
    const pos = App.mousePos(event);
    const oldScale = App.state.transform.scale;
    const factor = event.deltaY > 0 ? 0.9 : 1.1;
    const newScale = App.clamp(oldScale * factor, 0.05, 12);
    App.state.transform.x = pos.x - (pos.x - App.state.transform.x) * (newScale / oldScale);
    App.state.transform.y = pos.y - (pos.y - App.state.transform.y) * (newScale / oldScale);
    App.state.transform.scale = newScale;
  },

  handleContextMenu: (event) => {
    event.preventDefault();
    if (!App.state.image || !App.state.annotation) return;
    const pos = App.mousePos(event);
    App.state.contextImagePos = App.toImageCoords(pos.x, pos.y);
    App.showContextMenu(pos.x, pos.y);
  },

  showContextMenu: (x, y) => {
    const menu = document.getElementById("contextMenu");
    const items = document.getElementById("contextItems");
    const sideLabels = { left: "left", right: "right" };
    document.getElementById("contextTitle").textContent = "选择点位";
    items.innerHTML = "";
    App.state.schema.sides.forEach((side) => {
      App.state.schema.landmarks.forEach((landmark) => {
        const key = App.keyFor(side, landmark.name);
        const point = App.state.annotation.keypoints[key];
        const item = document.createElement("div");
        item.className = "context-item";
        item.innerHTML = `<span>${sideLabels[side] || side} #${landmark.number} ${landmark.label_zh}</span><small>${App.pointIsVisible(point) ? "已标" : ""}</small>`;
        item.addEventListener("click", (event) => {
          event.stopPropagation();
          App.placePoint(key, App.state.contextImagePos);
          App.hideContextMenu();
        });
        items.appendChild(item);
      });
    });
    const wrap = document.getElementById("canvasWrap").getBoundingClientRect();
    menu.style.left = `${App.clamp(x, 8, wrap.width - 250)}px`;
    menu.style.top = `${App.clamp(y, 8, wrap.height - 520)}px`;
    menu.style.display = "block";
  },

  hideContextMenu: () => {
    document.getElementById("contextMenu").style.display = "none";
  },

  handleKeyDown: (event) => {
    const tag = event.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || event.target.isContentEditable) return;
    if (!event.ctrlKey && !event.metaKey && !event.altKey && event.key === "ArrowLeft") {
      event.preventDefault();
      App.loadAdjacentImage(-1);
      return;
    }
    if (!event.ctrlKey && !event.metaKey && !event.altKey && event.key === "ArrowRight") {
      event.preventDefault();
      App.loadAdjacentImage(1);
      return;
    }
    if (event.code === "Space") {
      event.preventDefault();
      App.state.spaceHeld = true;
      document.getElementById("mainCanvas").style.cursor = "grab";
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      App.saveAnnotation();
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
      event.preventDefault();
      App.undo();
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
      event.preventDefault();
      App.redo();
      return;
    }
    if (event.key === "Delete") {
      if (App.state.selectedConnection) App.deleteSelectedConnection();
      else if (App.state.selectedShentonPoint) App.deleteSelectedShentonPoint();
      else if (App.state.selectedScanCorner !== null) App.deleteSelectedScanCorner();
      else if (App.state.selectedPoint) App.markMissing(App.state.selectedPoint);
    }
    if (event.key === "Escape") {
      App.state.pendingConnectionStart = null;
      App.state.selectedShentonPoint = null;
      App.state.selectedScanCorner = null;
      App.setStatus("已取消当前连线");
    }
    if (event.key === "Tab") {
      event.preventDefault();
      App.selectNextPoint();
    }
    if (event.key === "?") App.showShortcuts();
    if (event.key.toLowerCase() === "v") App.setTool("select");
    if (event.key.toLowerCase() === "p") App.setTool("point");
    if (event.key.toLowerCase() === "r") App.setTool("roi");
    if (event.key.toLowerCase() === "c") App.setTool("scan");
    if (event.key.toLowerCase() === "l") App.setTool("line");
    if (event.key.toLowerCase() === "s") App.setTool("shenton");
    if (event.key.toLowerCase() === "e") App.setImageView(App.state.imageView === "enhanced" ? "original" : "enhanced");
    if (event.key.toLowerCase() === "d") App.handleAutoDetectCurrent();
    if (event.key.toLowerCase() === "f") {
      if (App.state.activeTool === "scan" && App.currentScanTransform()) App.fitToScanTransform();
      else if (App.state.activeTool === "roi" && App.currentRoiCrop()) App.fitToRoi();
      else App.fitToScreen();
    }
    if (event.key.toLowerCase() === "h") {
      App.state.showLabels = !App.state.showLabels;
      App.syncDisplayToggles();
    }
  },

  handleKeyUp: (event) => {
    if (event.code === "Space") {
      App.state.spaceHeld = false;
      document.getElementById("mainCanvas").style.cursor =
        App.state.activeTool === "point" || App.state.activeTool === "line" || App.state.activeTool === "shenton" || App.state.activeTool === "roi"
          ? "crosshair"
          : "default";
    }
  },

  selectNextPoint: () => {
    const keys = App.allKeys();
    if (!keys.length) return;
    const index = keys.indexOf(App.state.selectedPoint);
    const next = keys[(index + 1 + keys.length) % keys.length];
    App.selectPoint(next);
    App.state.activePointKey = next;
  },

  allKeys: () => {
    const keys = [];
    App.state.schema.sides.forEach((side) => {
      App.state.schema.landmarks.forEach((landmark) => keys.push(App.keyFor(side, landmark.name)));
    });
    return keys;
  },

  mousePos: (event) => {
    const rect = document.getElementById("mainCanvas").getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  },

  toImageCoords: (x, y) => ({
    x: (x - App.state.transform.x) / App.state.transform.scale,
    y: (y - App.state.transform.y) / App.state.transform.scale,
  }),

  toViewportCoords: (x, y) => ({
    x: x * App.state.transform.scale + App.state.transform.x,
    y: y * App.state.transform.scale + App.state.transform.y,
  }),

  hitTestPoint: (x, y) => {
    let best = null;
    let bestDistance = Infinity;
    for (const [key, point] of Object.entries(App.state.annotation.keypoints || {})) {
      if (!App.pointIsDisplayed(key, point)) continue;
      const view = App.toViewportCoords(point.x, point.y);
      const distance = Math.hypot(view.x - x, view.y - y);
      if (distance < 12 && distance < bestDistance) {
        best = key;
        bestDistance = distance;
      }
    }
    return best;
  },

  hitTestShentonPoint: (x, y) => {
    if (!App.state.annotation) return null;
    App.normalizeShenton(App.state.annotation);
    let best = null;
    let bestDistance = Infinity;
    ["left", "right"].forEach((side) => {
      ["obturator_upper_curve", "femoral_neck_inner_lower_curve"].forEach((segmentKey) => {
        const points = App.state.annotation.shenton_curves[side]?.[segmentKey]?.points || [];
        points.forEach((point, index) => {
          const view = App.toViewportCoords(point.x, point.y);
          const distance = Math.hypot(view.x - x, view.y - y);
          if (distance < 12 && distance < bestDistance) {
            best = { side, segment: segmentKey, index };
            bestDistance = distance;
          }
        });
      });
    });
    return best;
  },

  hitTestScanCorner: (x, y) => {
    if (!App.state.annotation) return null;
    App.normalizeScanTransform(App.state.annotation);
    const corners = App.state.annotation.scan_transform?.corners || [];
    let best = null;
    let bestDistance = Infinity;
    corners.forEach((point, index) => {
      const view = App.toViewportCoords(point.x, point.y);
      const distance = Math.hypot(view.x - x, view.y - y);
      if (distance < 14 && distance < bestDistance) {
        best = index;
        bestDistance = distance;
      }
    });
    return best;
  },

  hitTestConnection: (x, y) => {
    let best = null;
    let bestDistance = Infinity;
    for (const connection of App.visibleConnections()) {
      const p1 = App.state.annotation.keypoints[connection.point_a];
      const p2 = App.state.annotation.keypoints[connection.point_b];
      if (!App.pointIsDisplayed(connection.point_a, p1) || !App.pointIsDisplayed(connection.point_b, p2)) continue;
      const a = App.toViewportCoords(p1.x, p1.y);
      const b = App.toViewportCoords(p2.x, p2.y);
      const distance = App.distanceToSegment(x, y, a.x, a.y, b.x, b.y);
      if (distance < 8 && distance < bestDistance) {
        best = connection.id;
        bestDistance = distance;
      }
    }
    return best;
  },

  distanceToSegment: (px, py, ax, ay, bx, by) => {
    const dx = bx - ax;
    const dy = by - ay;
    if (dx === 0 && dy === 0) return Math.hypot(px - ax, py - ay);
    const t = App.clamp(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0, 1);
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  },

  fitToScreen: () => {
    if (!App.state.image) return;
    const canvas = document.getElementById("mainCanvas");
    const rect = canvas.getBoundingClientRect();
    const scale = Math.min(rect.width / App.state.image.width, rect.height / App.state.image.height) * 0.92;
    App.state.transform = {
      x: (rect.width - App.state.image.width * scale) / 2,
      y: (rect.height - App.state.image.height * scale) / 2,
      scale,
    };
  },

  currentRoiCrop: () => {
    const roi = App.state.annotation?.roi_crop;
    if (!roi?.enabled || roi.x === null || roi.y === null || roi.width === null || roi.height === null) return null;
    return roi;
  },

  normalizedRoiBox: (start, end) => {
    if (!App.state.image) return null;
    const x1 = App.clamp(Math.min(start.x, end.x), 0, App.state.image.width);
    const y1 = App.clamp(Math.min(start.y, end.y), 0, App.state.image.height);
    const x2 = App.clamp(Math.max(start.x, end.x), 0, App.state.image.width);
    const y2 = App.clamp(Math.max(start.y, end.y), 0, App.state.image.height);
    const width = x2 - x1;
    const height = y2 - y1;
    if (width < 8 || height < 8) return null;
    return { x: x1, y: y1, width, height };
  },

  setRoiCrop: (box) => {
    if (!App.state.annotation || !box) return;
    App.state.annotation.roi_crop = {
      enabled: true,
      x: Number(box.x.toFixed(2)),
      y: Number(box.y.toFixed(2)),
      width: Number(box.width.toFixed(2)),
      height: Number(box.height.toFixed(2)),
      source: "manual",
      updated_at: new Date().toISOString(),
      annotator: document.getElementById("annotatorInput").value.trim() || "default",
    };
    App.renderRoiPanel();
  },

  clearRoiCrop: () => {
    if (!App.state.annotation) return;
    App.pushHistory();
    App.state.annotation.roi_crop = {
      enabled: false,
      x: null,
      y: null,
      width: null,
      height: null,
      source: "manual",
      updated_at: new Date().toISOString(),
      annotator: document.getElementById("annotatorInput").value.trim() || "default",
    };
    App.state.roiDraft = null;
    App.renderRoiPanel();
    App.pushHistory();
    App.scheduleAutosave();
    App.setStatus("ROI 已清空；识别将使用扫描四角或全图");
  },

  fitToRoi: () => {
    const roi = App.currentRoiCrop();
    if (!roi) {
      App.fitToScreen();
      App.setStatus("当前没有 ROI，已适配全图");
      return;
    }
    const canvas = document.getElementById("mainCanvas");
    const rect = canvas.getBoundingClientRect();
    const scale = Math.min(rect.width / roi.width, rect.height / roi.height) * 0.88;
    App.state.transform = {
      x: (rect.width - roi.width * scale) / 2 - roi.x * scale,
      y: (rect.height - roi.height * scale) / 2 - roi.y * scale,
      scale,
    };
  },

  renderRoiPanel: () => {
    const target = document.getElementById("roiHintText");
    if (!target) return;
    const roi = App.currentRoiCrop();
    if (!roi) {
      target.textContent = "拖拽画出需要关注的 X 光区域；ROI 启用时识别优先使用 ROI，未启用 ROI 时才使用扫描四角校正。";
      return;
    }
    target.textContent = `已启用 ROI：x ${roi.x.toFixed(0)}, y ${roi.y.toFixed(0)}, ${roi.width.toFixed(0)} × ${roi.height.toFixed(0)}；识别会优先使用 ROI。`;
  },

  currentScanTransform: () => {
    const scan = App.state.annotation?.scan_transform;
    if (!scan || !Array.isArray(scan.corners) || scan.corners.length !== 4 || !scan.enabled) return null;
    return scan;
  },

  scanPolygonArea: (corners) => {
    if (!Array.isArray(corners) || corners.length < 3) return 0;
    let area = 0;
    corners.forEach((point, index) => {
      const next = corners[(index + 1) % corners.length];
      area += point.x * next.y - next.x * point.y;
    });
    return Math.abs(area) / 2;
  },

  orderScanCorners: (corners) => {
    if (!Array.isArray(corners) || corners.length !== 4) return corners || [];
    const metrics = corners.map((point, index) => ({
      point,
      index,
      sum: point.x + point.y,
      diff: point.y - point.x,
    }));
    const pick = (sorter) => [...metrics].sort(sorter)[0];
    const ordered = [
      pick((a, b) => a.sum - b.sum),
      pick((a, b) => a.diff - b.diff),
      pick((a, b) => b.sum - a.sum),
      pick((a, b) => b.diff - a.diff),
    ];
    if (new Set(ordered.map((item) => item.index)).size !== 4) return corners;
    return ordered.map((item) => ({ x: item.point.x, y: item.point.y }));
  },

  addScanCorner: (imagePos) => {
    if (!App.state.annotation || !App.state.image) return;
    App.normalizeScanTransform(App.state.annotation);
    const scan = App.state.annotation.scan_transform;
    if (scan.corners.length >= 4) {
      App.setStatus("扫描四角已完整，可拖拽调整或清空重画");
      return;
    }
    App.pushHistory();
    scan.corners.push({
      x: Number(App.clamp(imagePos.x, 0, App.state.image.width).toFixed(2)),
      y: Number(App.clamp(imagePos.y, 0, App.state.image.height).toFixed(2)),
    });
    if (scan.corners.length === 4) {
      scan.corners = App.orderScanCorners(scan.corners);
      scan.enabled = App.scanPolygonArea(scan.corners) >= 64;
    }
    scan.updated_at = new Date().toISOString();
    scan.annotator = document.getElementById("annotatorInput").value.trim() || "default";
    App.state.selectedScanCorner = scan.corners.length - 1;
    App.renderScanPanel();
    App.updateSelectedBox();
    App.pushHistory();
    App.scheduleAutosave();
  },

  clearScanTransform: () => {
    if (!App.state.annotation) return;
    App.pushHistory();
    App.state.annotation.scan_transform = {
      enabled: false,
      corners: [],
      mode: "manual_four_corners",
      source: "manual",
      updated_at: new Date().toISOString(),
      annotator: document.getElementById("annotatorInput").value.trim() || "default",
    };
    App.state.selectedScanCorner = null;
    App.renderScanPanel();
    App.updateSelectedBox();
    App.pushHistory();
    App.scheduleAutosave();
    App.setStatus("扫描四角已清空");
  },

  deleteSelectedScanCorner: () => {
    if (!App.state.annotation || App.state.selectedScanCorner === null) return;
    App.normalizeScanTransform(App.state.annotation);
    const scan = App.state.annotation.scan_transform;
    if (!scan.corners[App.state.selectedScanCorner]) return;
    App.pushHistory();
    scan.corners.splice(App.state.selectedScanCorner, 1);
    scan.enabled = false;
    scan.updated_at = new Date().toISOString();
    App.state.selectedScanCorner = null;
    App.renderScanPanel();
    App.updateSelectedBox();
    App.pushHistory();
    App.scheduleAutosave();
  },

  fitToScanTransform: () => {
    const scan = App.currentScanTransform();
    if (!scan) {
      App.fitToScreen();
      App.setStatus("当前没有完整扫描四角，已适配全图");
      return;
    }
    const xs = scan.corners.map((point) => point.x);
    const ys = scan.corners.map((point) => point.y);
    const box = {
      x: Math.min(...xs),
      y: Math.min(...ys),
      width: Math.max(...xs) - Math.min(...xs),
      height: Math.max(...ys) - Math.min(...ys),
    };
    const canvas = document.getElementById("mainCanvas");
    const rect = canvas.getBoundingClientRect();
    const scale = Math.min(rect.width / box.width, rect.height / box.height) * 0.88;
    App.state.transform = {
      x: (rect.width - box.width * scale) / 2 - box.x * scale,
      y: (rect.height - box.height * scale) / 2 - box.y * scale,
      scale,
    };
  },

  renderScanPanel: () => {
    const target = document.getElementById("scanHintText");
    if (!target || !App.state.annotation) return;
    App.normalizeScanTransform(App.state.annotation);
    const scan = App.state.annotation.scan_transform;
    if (scan.enabled) {
      target.textContent = "扫描四角已启用；未启用 ROI 时，识别会先透视校正再映射回原图坐标。";
      return;
    }
    const next = SCAN_CORNER_LABELS[scan.corners.length] || "完成";
    target.textContent = `依次点击片子左上、右上、右下、左下四个角；当前 ${scan.corners.length}/4，下一个：${next}。`;
  },

  drawLoop: () => {
    App.draw();
    requestAnimationFrame(App.drawLoop);
  },

  draw: () => {
    const canvas = document.getElementById("mainCanvas");
    const rect = canvas.getBoundingClientRect();
    const ctx = canvas.getContext("2d");
    ctx.save();
    ctx.setTransform(window.devicePixelRatio || 1, 0, 0, window.devicePixelRatio || 1, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#1b1d1f";
    ctx.fillRect(0, 0, rect.width, rect.height);
    if (!App.state.image) {
      ctx.fillStyle = "#c7d0d5";
      ctx.font = "15px sans-serif";
      ctx.fillText("打开图像后开始标注", 24, 34);
      ctx.restore();
      return;
    }

    ctx.translate(App.state.transform.x, App.state.transform.y);
    ctx.scale(App.state.transform.scale, App.state.transform.scale);
    ctx.drawImage(App.state.image, 0, 0);
    App.drawRoiCrop(ctx);
    App.drawScanTransform(ctx);
    App.drawConnections(ctx);
    App.drawShentonCurves(ctx);
    App.drawMeasurementLines(ctx);
    App.drawPoints(ctx);
    ctx.restore();
  },

  visibleConnections: () =>
    (App.state.annotation?.connections || []).filter((item) => {
      if (item.visible === false) return false;
      if (item.source === "default") return App.state.showDefaultConnections;
      if (item.source === "manual") return App.state.showManualConnections;
      return true;
    }),

  drawRoiCrop: (ctx) => {
    const roi = App.state.roiDraft || App.currentRoiCrop();
    if (!roi) return;
    const inv = 1 / App.state.transform.scale;
    ctx.save();
    ctx.strokeStyle = App.state.roiDraft ? "#ffd43b" : "#40c057";
    ctx.lineWidth = 2.4 * inv;
    ctx.setLineDash([8 * inv, 5 * inv]);
    ctx.strokeRect(roi.x, roi.y, roi.width, roi.height);
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(64, 192, 87, 0.08)";
    ctx.fillRect(roi.x, roi.y, roi.width, roi.height);
    ctx.font = `${13 * inv}px sans-serif`;
    ctx.lineWidth = 3 * inv;
    const label = App.state.roiDraft ? "ROI" : "ROI 识别区域";
    ctx.strokeStyle = "rgba(0,0,0,0.75)";
    ctx.strokeText(label, roi.x + 8 * inv, Math.max(14 * inv, roi.y - 8 * inv));
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, roi.x + 8 * inv, Math.max(14 * inv, roi.y - 8 * inv));
    ctx.restore();
  },

  drawScanTransform: (ctx) => {
    const scan = App.state.annotation?.scan_transform;
    if (!scan || !Array.isArray(scan.corners) || !scan.corners.length) return;
    const inv = 1 / App.state.transform.scale;
    const corners = scan.corners;
    ctx.save();
    if (corners.length > 1) {
      ctx.beginPath();
      ctx.moveTo(corners[0].x, corners[0].y);
      corners.slice(1).forEach((point) => ctx.lineTo(point.x, point.y));
      if (corners.length === 4) ctx.closePath();
      ctx.strokeStyle = scan.enabled ? "#15aabf" : "#ffd43b";
      ctx.lineWidth = 2.5 * inv;
      ctx.setLineDash(scan.enabled ? [] : [7 * inv, 4 * inv]);
      ctx.stroke();
      ctx.setLineDash([]);
      if (scan.enabled) {
        ctx.fillStyle = "rgba(21, 170, 191, 0.08)";
        ctx.fill();
      }
    }
    corners.forEach((point, index) => {
      const selected = App.state.selectedScanCorner === index;
      ctx.beginPath();
      ctx.arc(point.x, point.y, (selected ? 6.2 : 4.8) * inv, 0, Math.PI * 2);
      ctx.fillStyle = selected ? "#ffffff" : "#15aabf";
      ctx.fill();
      ctx.lineWidth = (selected ? 2.8 : 1.7) * inv;
      ctx.strokeStyle = "#083344";
      ctx.stroke();
      ctx.font = `${12 * inv}px sans-serif`;
      ctx.lineWidth = 3 * inv;
      const label = SCAN_CORNER_LABELS[index] || `${index + 1}`;
      ctx.strokeStyle = "rgba(0, 0, 0, 0.72)";
      ctx.strokeText(label, point.x + 8 * inv, point.y - 8 * inv);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, point.x + 8 * inv, point.y - 8 * inv);
    });
    ctx.restore();
  },

  drawConnections: (ctx) => {
    if (!App.state.annotation) return;
    const inv = 1 / App.state.transform.scale;
    App.visibleConnections().forEach((connection) => {
      const p1 = App.state.annotation.keypoints[connection.point_a];
      const p2 = App.state.annotation.keypoints[connection.point_b];
      if (!App.pointIsDisplayed(connection.point_a, p1) || !App.pointIsDisplayed(connection.point_b, p2)) return;
      const selected = connection.id === App.state.selectedConnection;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.strokeStyle = selected ? "#ffffff" : App.connectionColor(connection);
      ctx.lineWidth = (selected ? 3 : 1.7) * inv;
      if (connection.source === "manual") ctx.setLineDash([7 * inv, 4 * inv]);
      ctx.stroke();
      ctx.setLineDash([]);
    });
  },

  drawShentonCurves: (ctx) => {
    if (!App.state.annotation || !App.state.showShenton) return;
    App.normalizeShenton(App.state.annotation);
    const inv = 1 / App.state.transform.scale;
    ["left", "right"].forEach((side) => {
      ["obturator_upper_curve", "femoral_neck_inner_lower_curve"].forEach((segmentKey) => {
        const points = App.state.annotation.shenton_curves[side]?.[segmentKey]?.points || [];
        if (!points.length) return;
        const color = App.shentonColor(side, segmentKey);
        if (points.length > 1) {
          ctx.beginPath();
          ctx.moveTo(points[0].x, points[0].y);
          points.slice(1).forEach((point) => ctx.lineTo(point.x, point.y));
          ctx.strokeStyle = color;
          ctx.lineWidth = 2.2 * inv;
          ctx.setLineDash(segmentKey === "obturator_upper_curve" ? [] : [7 * inv, 4 * inv]);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        points.forEach((point, index) => {
          const selected =
            App.state.selectedShentonPoint?.side === side &&
            App.state.selectedShentonPoint?.segment === segmentKey &&
            App.state.selectedShentonPoint?.index === index;
          ctx.beginPath();
          ctx.arc(point.x, point.y, (selected ? 5.8 : 4.3) * inv, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
          ctx.lineWidth = (selected ? 2.8 : 1.8) * inv;
          ctx.strokeStyle = selected ? "#ffffff" : "rgba(0,0,0,0.72)";
          ctx.stroke();
        });
      });
    });
  },

  drawMeasurementLines: (ctx) => {
    if (!App.state.annotation || !App.state.showMeasurements) return;
    const inv = 1 / App.state.transform.scale;
    ["left", "right"].forEach((side) => {
      const pair = App.closestShentonEndpoints(side);
      if (!pair) return;
      ctx.beginPath();
      ctx.moveTo(pair.a.x, pair.a.y);
      ctx.lineTo(pair.b.x, pair.b.y);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.lineWidth = 1.5 * inv;
      ctx.setLineDash([3 * inv, 5 * inv]);
      ctx.stroke();
      ctx.setLineDash([]);
    });
  },

  closestShentonEndpoints: (side) => {
    const sideCurves = App.state.annotation?.shenton_curves?.[side];
    const obturator = sideCurves?.obturator_upper_curve?.points || [];
    const femoral = sideCurves?.femoral_neck_inner_lower_curve?.points || [];
    if (obturator.length < 1 || femoral.length < 1) return null;
    const endpointsA = [obturator[0], obturator[obturator.length - 1]];
    const endpointsB = [femoral[0], femoral[femoral.length - 1]];
    let best = null;
    let bestDistance = Infinity;
    endpointsA.forEach((a) => {
      endpointsB.forEach((b) => {
        const distance = Math.hypot(a.x - b.x, a.y - b.y);
        if (distance < bestDistance) {
          best = { a, b };
          bestDistance = distance;
        }
      });
    });
    return best;
  },

  shentonColor: (side, segmentKey) => {
    if (segmentKey === "obturator_upper_curve") return side === "left" ? "#ff922b" : "#4dabf7";
    return side === "left" ? "#ffd43b" : "#63e6be";
  },

  drawPoints: (ctx) => {
    if (!App.state.annotation) return;
    const inv = 1 / App.state.transform.scale;
    App.allKeys().forEach((key) => {
      const point = App.state.annotation.keypoints[key];
      if (!App.pointIsDisplayed(key, point)) return;
      const color = point.side === "left" ? "#d9480f" : "#1c7ed6";
      const pending = key === App.state.pendingConnectionStart;
      ctx.beginPath();
      ctx.arc(point.x, point.y, (pending ? 7 : 5.2) * inv, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = (pending || key === App.state.selectedPoint ? 3 : 2) * inv;
      ctx.strokeStyle = pending || key === App.state.selectedPoint ? "#ffffff" : App.sourceColor(point.source);
      ctx.stroke();
      if (App.state.showLabels) {
        ctx.font = `${12 * inv}px sans-serif`;
        ctx.lineWidth = 3 * inv;
        const text = `#${point.number}`;
        ctx.strokeStyle = "rgba(0,0,0,0.75)";
        ctx.strokeText(text, point.x + 8 * inv, point.y - 7 * inv);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(text, point.x + 8 * inv, point.y - 7 * inv);
      }
    });
  },

  connectionColor: (connection) => {
    if (connection.source === "manual") return "#ffd43b";
    if (connection.label && connection.label.includes("图像左")) return "#63e6be";
    return "#ff8a8a";
  },

  sourceColor: (source) => {
    if (source === "retuve") return "#6741d9";
    if (source === "manual") return "#2f9e44";
    if (App.isEstimatedSource(source)) return "#e67700";
    return "#adb5bd";
  },

  refreshPointList: () => {
    document.querySelectorAll(".point-row").forEach((row) => {
      const key = row.dataset.key;
      const point = App.state.annotation?.keypoints?.[key];
      const visible = App.pointIsVisible(point);
      row.classList.toggle("missing", !visible);
      row.classList.toggle("selected", key === App.state.selectedPoint);
      const checkbox = row.querySelector("input");
      checkbox.checked = visible;
      const badge = row.querySelector(".source-badge");
      const source = visible ? point.source || "manual" : "missing";
      badge.className = `source-badge ${App.sourceClass(source)}`;
      badge.textContent = App.sourceLabel(source);
    });
  },

  renderConnectionList: () => {
    const list = document.getElementById("connectionList");
    list.innerHTML = "";
    if (!App.state.annotation) return;
    App.visibleConnections().forEach((connection) => {
      const row = document.createElement("div");
      row.className = `connection-row ${connection.id === App.state.selectedConnection ? "selected" : ""}`;
      row.innerHTML = `<span>${App.connectionLabel(connection)}</span><small>${connection.source === "manual" ? "人工" : "默认"}</small>`;
      row.addEventListener("click", () => App.selectConnection(connection.id));
      list.appendChild(row);
    });
  },

  sourceLabel: (source) => {
    if (source === "retuve") return "retuve";
    if (source === "manual") return "人工";
    if (source === "imported_label") return "导入";
    if (source === "pose11_side") return "模型";
    if (source === "template_guess") return "模板";
    if (App.isEstimatedSource(source)) return "估计";
    return "缺失";
  },

  sourceClass: (source) => {
    if (App.isEstimatedSource(source)) return "estimated";
    return source || "missing";
  },

  isEstimatedSource: (source) =>
    source === "estimated" ||
    source === "template_guess" ||
    source === "hippelvis_like_mask" ||
    source === "imported_label" ||
    source === "pose11_side",

  updateSummary: () => {
    const counts = { model: 0, estimated: 0, manual: 0, missing: 0 };
    if (!App.state.annotation) {
      counts.missing = 22;
    } else {
      Object.values(App.state.annotation.keypoints).forEach((point) => {
        if (!App.pointIsVisible(point)) counts.missing += 1;
        else if (point.source === "retuve" || point.source === "pose11_side") counts.model += 1;
        else if (point.source === "manual") counts.manual += 1;
        else counts.estimated += 1;
      });
    }
    const complete = 22 - counts.missing;
    document.getElementById("completionText").textContent = `${complete} / 22`;
    document.getElementById("countModel").textContent = counts.model;
    document.getElementById("countEstimated").textContent = counts.estimated;
    document.getElementById("countManual").textContent = counts.manual;
    document.getElementById("countMissing").textContent = counts.missing;
  },

  renderKeypointConfirmation: () => {
    const text = document.getElementById("keypointConfirmText");
    const button = document.getElementById("btnConfirmKeypointsComplete");
    if (!text || !button) return;
    if (!App.state.annotation) {
      text.textContent = "关键点尚未人工确认完成";
      button.disabled = true;
      return;
    }
    button.disabled = false;
    const review = App.state.annotation.review?.manual_keypoints_complete;
    if (review?.status === "confirmed") {
      const annotator = review.annotator || "default";
      const timeText = review.updated_at ? new Date(review.updated_at).toLocaleString() : "";
      text.textContent = `已由 ${annotator} 确认关键点完成${timeText ? `（${timeText}）` : ""}`;
      return;
    }
    const visible = App.visibleKeypointCount();
    text.textContent = visible >= 22 ? "22 个关键点已齐，等待人工确认完成" : `已标 ${visible}/22，未完成`;
  },

  renderShentonConfirmation: () => {
    const text = document.getElementById("shentonConfirmText");
    const button = document.getElementById("btnConfirmShentonComplete");
    if (!text || !button) return;
    if (!App.state.annotation) {
      text.textContent = "沈通线尚未人工确认完成";
      button.disabled = true;
      return;
    }
    button.disabled = false;
    const confirmed = App.shentonConfirmed();
    if (confirmed) {
      const meta = App.state.annotation.review?.manual_shenton_complete || {};
      const annotator = meta.annotator || "default";
      const timeText = meta.updated_at ? new Date(meta.updated_at).toLocaleString() : "";
      text.textContent = `已由 ${annotator} 确认沈通线完成${timeText ? `（${timeText}）` : ""}`;
      return;
    }
    const shenton = App.state.annotation.shenton_curves || {};
    const review = App.state.annotation.shenton_review || {};
    let completeSides = 0;
    for (const side of ["left", "right"]) {
      const curves = shenton[side] || {};
      let curvesOk = true;
      for (const seg of ["obturator_upper_curve", "femoral_neck_inner_lower_curve"]) {
        if ((curves[seg]?.points || []).length < 3) { curvesOk = false; break; }
      }
      const sideReview = review[side]?.status || "not_reviewed";
      if (curvesOk && ["continuous", "discontinuous", "uncertain"].includes(sideReview)) completeSides += 1;
    }
    text.textContent = completeSides >= 2 ? "两侧沈通线已标注，等待人工确认完成" : `沈通线 ${completeSides}/2 侧完成`;
  },

  updateSelectedBox: () => {
    const label = document.getElementById("selectedLabel");
    const coords = document.getElementById("selectedCoords");
    const connectionLabel = document.getElementById("selectedConnectionLabel");
    if (App.state.annotation && App.state.selectedShentonPoint) {
      const selected = App.state.selectedShentonPoint;
      const point = App.state.annotation.shenton_curves?.[selected.side]?.[selected.segment]?.points?.[selected.index];
      label.textContent = `${selected.side === "left" ? "左" : "右"} Shenton ${selected.segment === "obturator_upper_curve" ? "闭孔上缘" : "股骨颈内下缘"}`;
      coords.textContent = point ? `${point.x.toFixed(1)}, ${point.y.toFixed(1)}` : "-";
      connectionLabel.textContent = "未选中连线";
      return;
    }
    if (App.state.annotation && App.state.selectedScanCorner !== null) {
      const point = App.state.annotation.scan_transform?.corners?.[App.state.selectedScanCorner];
      label.textContent = `扫描四角 ${SCAN_CORNER_LABELS[App.state.selectedScanCorner] || App.state.selectedScanCorner + 1}`;
      coords.textContent = point ? `${point.x.toFixed(1)}, ${point.y.toFixed(1)}` : "-";
      connectionLabel.textContent = "未选中连线";
      return;
    }
    if (!App.state.annotation || !App.state.selectedPoint) {
      label.textContent = "-";
      coords.textContent = App.state.activeTool === "line" && App.state.pendingConnectionStart ? "请选择第二个点" : "-";
    } else {
      const point = App.state.annotation.keypoints[App.state.selectedPoint];
      label.textContent = point.label;
      coords.textContent = App.pointIsVisible(point) ? `${point.x.toFixed(1)}, ${point.y.toFixed(1)} · ${App.sourceLabel(point.source)}` : "缺失";
    }
    const selectedConnection = App.state.annotation?.connections?.find((item) => item.id === App.state.selectedConnection);
    connectionLabel.textContent = selectedConnection ? App.connectionLabel(selectedConnection) : "未选中连线";
  },

  connectionLabel: (connection) => {
    const a = App.state.annotation?.keypoints?.[connection.point_a];
    const b = App.state.annotation?.keypoints?.[connection.point_b];
    if (!a || !b) return connection.label || "连线";
    const sideA = a.side === "left" ? "左" : "右";
    const sideB = b.side === "left" ? "左" : "右";
    return `${sideA}#${a.number} - ${sideB}#${b.number}`;
  },

  renderWarnings: () => {
    const container = document.getElementById("warningList");
    container.innerHTML = "";
    const warnings = [
      ...(App.state.annotation?.auto_initialization?.warnings || []),
      ...(App.state.annotation?.image?.dicom_warnings || []),
    ];
    warnings.forEach((warning) => {
      const item = document.createElement("div");
      item.className = "warning-item";
      item.textContent = warning;
      container.appendChild(item);
    });
  },

  clamp: (value, min, max) => Math.max(min, Math.min(max, value)),
};

window.addEventListener("load", App.init);
