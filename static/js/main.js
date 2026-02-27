
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// --- Global Variables ---
let scene, camera, renderer, controls;
let modelPivot; // Group to hold the loaded model
let gpuInterval;
let selectedSamplePath = null; // Track selected sample
let initialModelRotation = new THREE.Euler(0, 0, 0);
const STEP_SIZE = 0.02;
const SCALE_STEP = 0.05;
const MIN_SCALE = 0.3;
const MAX_SCALE = 3.0;
const HOLD_INTERVAL_MS = 90;

// --- DOM Elements ---
let canvasContainer;
let consoleOutput;
let dropZone;
let imageInput;
let previewImage;
let generateBtn;
let gpu0Bar;
let gpu0Val;
let gpu1Bar;
let gpu1Val;
let holographBtn;
let rotUpBtn;
let rotDownBtn;
let rotLeftBtn;
let rotRightBtn;
let rotResetBtn;
let zoomInBtn;
let zoomOutBtn;
let activeHoldInterval = null;

// --- Initialization ---
window.onload = () => {
    try {
        // 1. Protocol Check
        if (location.protocol !== 'https:' && location.hostname !== 'localhost') {
            const warningElement = document.getElementById('protocol-warning');
            if (warningElement) warningElement.style.display = 'block';
        }

        // 2. WebXR Shim
        if (typeof LookingGlassWebXRPolyfill !== 'undefined') {
            new LookingGlassWebXRPolyfill();
        }

        initDomElements();
        initThreeJS();
        initWebXR();
        startGPUMonitoring();
        setupEventListeners();
    } catch (error) {
        console.error("Initialization error:", error);
        log("System initialization failed. Check console for details.", "error");
    }
};

function initDomElements() {
    canvasContainer = document.getElementById('canvas-container');
    consoleOutput = document.getElementById('console-output');
    dropZone = document.getElementById('drop-zone');
    imageInput = document.getElementById('imageInput');
    previewImage = document.getElementById('preview-image');
    generateBtn = document.getElementById('generate-btn');
    gpu0Bar = document.getElementById('gpu0-bar');
    gpu0Val = document.getElementById('gpu0-val');
    gpu1Bar = document.getElementById('gpu1-bar');
    gpu1Val = document.getElementById('gpu1-val');
    holographBtn = document.getElementById('view-hologram-btn');
    rotUpBtn = document.getElementById('rot-up');
    rotDownBtn = document.getElementById('rot-down');
    rotLeftBtn = document.getElementById('rot-left');
    rotRightBtn = document.getElementById('rot-right');
    rotResetBtn = document.getElementById('rot-reset');
    zoomInBtn = document.getElementById('zoom-in');
    zoomOutBtn = document.getElementById('zoom-out');
}

// --- Three.js Setup ---
function initThreeJS() {
    log("Initializing 3D Viewport...");
    
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a); // Dark gray background

    // Center camera
    camera = new THREE.PerspectiveCamera(35, canvasContainer.clientWidth / canvasContainer.clientHeight, 0.1, 100);
    camera.position.set(0, 1.5, 4);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.shadowMap.enabled = true;
    renderer.xr.enabled = true; // Enable WebXR
    canvasContainer.appendChild(renderer.domElement);

    // Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 0, 0);

    const ambientLight = new THREE.AmbientLight(0xffffff, 1.5);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 2.0);
    directionalLight.position.set(0, 10, 10);
    scene.add(directionalLight);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(2, 4, 3);
    dirLight.castShadow = true;
    scene.add(dirLight);
    
    const backLight = new THREE.DirectionalLight(0xffffff, 0.5);
    backLight.position.set(-2, 2, -3);
    scene.add(backLight);

    // Geometry Placeholder (Pivot)
    modelPivot = new THREE.Group();
    scene.add(modelPivot);

    // Grid Helper (optional visual aid)
    // const gridHelper = new THREE.GridHelper(2, 20, 0x444444, 0x222222);
    // scene.add(gridHelper);

    // Listen to resize
    window.addEventListener('resize', onWindowResize);

    // Animation Loop
    renderer.setAnimationLoop(render);
    
    log("Viewport Ready.");
}

function initWebXR() {
    log("Initializing Looking Glass WebXR...");
    // Configure Looking Glass using the global object from the UMD bundle
    if (typeof LookingGlassWebXRPolyfill !== 'undefined') {
        LookingGlassConfig.tileHeight = 512;
        LookingGlassConfig.numViews = 45;
        new LookingGlassWebXRPolyfill();
    } else {
        log("Looking Glass WebXR library not found. Holographic mode disabled.", "error");
    }
    
    holographBtn.addEventListener('click', async () => {
         try {
            await document.body.requestFullscreen(); // Often required for WebXR
             // This effectively enters the VR session provided by the polyfill
             // The polyfill overrides navigator.xr.requestSession
             const session = await navigator.xr.requestSession('immersive-vr');
             renderer.xr.setSession(session);
             log("Entering Holographic Mode...");
         } catch (e) {
             console.error(e);
             log("Error entering Holographic Mode: " + e.message, "error");
             alert("Could not enter Holographic Mode. Ensure Looking Glass Bridge is running.");
         }
    });

}

function render(time) {
    controls.update();
    renderer.render(scene, camera);
}

function onWindowResize() {
    camera.aspect = canvasContainer.clientWidth / canvasContainer.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
}

function renderNow() {
    if (!renderer || !scene || !camera) return;
    renderer.render(scene, camera);
}

// --- Interaction Logic ---
function setupEventListeners() {
    // Model Selector
    const modelSelect = document.getElementById('model-select');
    if (modelSelect) {
        modelSelect.addEventListener('change', (e) => {
            log(`AI Engine switched to: ${e.target.value}`, "system");
        });
    }

    // Generate Button
    generateBtn.addEventListener('click', handleGenerate);
    
    // Click to Upload
    dropZone.addEventListener('click', (e) => {
        imageInput.click();
    });

    imageInput.addEventListener('click', (e) => {
        e.stopPropagation();
    });

    // File Input Change
    imageInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    // Drag & Drop Events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop zone
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('active');
    }

    function unhighlight(e) {
        dropZone.classList.remove('active');
    }

    // Handle Drop
    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    // Sample Gallery Logic
    const sampleItems = document.querySelectorAll('.sample-item');
    sampleItems.forEach(item => {
        item.addEventListener('click', (e) => {
            // Clear manual uploads
            window.selectedFile = null;
            imageInput.value = '';
            previewImage.classList.add('hidden');
            dropZone.querySelector('p').innerHTML = `Drag & Drop Image or <span class="action-text">Click to Upload</span>`;
            dropZone.querySelector('i').classList.remove('hidden');

            // Update selection state
            sampleItems.forEach(s => s.classList.remove('selected'));
            e.target.classList.add('selected');
            
            selectedSamplePath = e.target.getAttribute('data-path');
            log(`Sample selected: ${selectedSamplePath}`);
        });
    });

    bindHoldAction(rotLeftBtn, () => fineTuneRotation('y', -1, rotLeftBtn));
    bindHoldAction(rotRightBtn, () => fineTuneRotation('y', 1, rotRightBtn));
    bindHoldAction(rotUpBtn, () => fineTuneRotation('x', -1, rotUpBtn));
    bindHoldAction(rotDownBtn, () => fineTuneRotation('x', 1, rotDownBtn));

    if (rotResetBtn) {
        rotResetBtn.addEventListener('click', () => resetModelRotation(rotResetBtn));
    }

    bindHoldAction(zoomInBtn, () => fineTuneZoom(1, zoomInBtn));
    bindHoldAction(zoomOutBtn, () => fineTuneZoom(-1, zoomOutBtn));

    document.addEventListener('pointerup', stopHoldAction);
    document.addEventListener('pointercancel', stopHoldAction);
}

function bindHoldAction(button, action) {
    if (!button) return;

    button.addEventListener('pointerdown', (event) => {
        event.preventDefault();
        stopHoldAction();
        action();
        activeHoldInterval = setInterval(action, HOLD_INTERVAL_MS);
    });

    button.addEventListener('pointerleave', stopHoldAction);
    button.addEventListener('pointerup', stopHoldAction);
    button.addEventListener('pointercancel', stopHoldAction);
}

function stopHoldAction() {
    if (activeHoldInterval) {
        clearInterval(activeHoldInterval);
        activeHoldInterval = null;
    }
}

function getRotationDeg() {
    return {
        x: THREE.MathUtils.radToDeg(modelPivot.rotation.x),
        y: THREE.MathUtils.radToDeg(modelPivot.rotation.y),
        z: THREE.MathUtils.radToDeg(modelPivot.rotation.z)
    };
}

function flashButton(button) {
    if (!button) return;
    button.classList.add('active');
    setTimeout(() => button.classList.remove('active'), 120);
}

function fineTuneRotation(axis, direction, button = null) {
    if (!modelPivot || modelPivot.children.length === 0) {
        log('No model loaded. Generate a model first to use precision controls.', 'error');
        flashButton(button);
        return;
    }

    modelPivot.rotation[axis] += STEP_SIZE * direction;
    controls.update();
    renderNow();
    flashButton(button);

    const rotation = getRotationDeg();
    const axisLabel = axis.toUpperCase();
    const delta = `${direction > 0 ? '+' : '-'}${STEP_SIZE.toFixed(2)}`;
    log(`[SYSTEM] Fine-tuning: Rotation ${axisLabel} ${delta} | X=${rotation.x.toFixed(1)}°, Y=${rotation.y.toFixed(1)}°`, 'system');
}

function fineTuneZoom(direction, button = null) {
    if (!modelPivot || modelPivot.children.length === 0) {
        log('No model loaded. Zoom controls are unavailable.', 'error');
        flashButton(button);
        return;
    }

    const nextScale = THREE.MathUtils.clamp(modelPivot.scale.x + (direction * SCALE_STEP), MIN_SCALE, MAX_SCALE);
    modelPivot.scale.set(nextScale, nextScale, nextScale);
    controls.update();
    renderNow();
    flashButton(button);
    const delta = `${direction > 0 ? '+' : '-'}${SCALE_STEP.toFixed(2)}`;
    log(`[SYSTEM] Fine-tuning: Scale ${delta} | Current=${nextScale.toFixed(2)}`, 'system');
}

function resetModelRotation(button = null) {
    if (!modelPivot || modelPivot.children.length === 0) {
        log('No model loaded. Reset is unavailable.', 'error');
        flashButton(button);
        return;
    }

    modelPivot.rotation.copy(initialModelRotation);
    controls.target.set(0, 0, 0);
    controls.update();
    renderNow();
    flashButton(button);
    log('[SYSTEM] Fine-tuning: Reset to initial orientation.', 'success');
}

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            // Clear sample selection
            selectedSamplePath = null;
            document.querySelectorAll('.sample-item').forEach(s => s.classList.remove('selected'));

            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewImage.classList.remove('hidden');
                
                // Show file name
                const p = dropZone.querySelector('p');
                p.classList.remove('hidden');
                p.innerHTML = `Selected: <strong>${file.name}</strong>`;
                
                // Hide icon but keep text visible
                dropZone.querySelector('i').classList.add('hidden');
            };
            reader.readAsDataURL(file);
            log(`Image selected: ${file.name}`);
            
            window.selectedFile = file; 
        } else {
            log("Invalid file type. Please upload an image.", "error");
        }
    }
}

async function handleGenerate() {
    const fileToUpload = window.selectedFile || imageInput.files[0];
    
    if (!fileToUpload && !selectedSamplePath) {
        log("No image selected! Please upload an image or select a sample first.", "error");
        return;
    }

    const formData = new FormData();
    let logMessage = "";
    
    const engineSelect = document.getElementById('model-select');
    const engineType = engineSelect ? engineSelect.value : 'triposr';
    formData.append('engine_type', engineType);

    if (fileToUpload) {
        formData.append('file', fileToUpload);
        logMessage = `Initiating generation for ${fileToUpload.name} using ${engineType}...`;
    } else if (selectedSamplePath) {
        formData.append('sample_path', selectedSamplePath);
        logMessage = `Initiating generation for sample ${selectedSamplePath} using ${engineType}...`;
    }

    log(logMessage, "system");
    generateBtn.disabled = true;
    generateBtn.textContent = "Processing... (This may take 15-30s)";
    generateBtn.classList.add('pulse');

    try {
        const response = await fetch('/generate', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();
        log("Generation Complete! Loading Asset...", "success");
        
        loadModel(data.asset_url);

    } catch (error) {
        log(`Generation Failed: ${error.message}`, "error");
    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = "INITIALIZE GENERATION SEQUENCE";
        generateBtn.classList.remove('pulse');
    }
}

function loadModel(url) {
    const loader = new GLTFLoader();
    
    // Clear previous model
    while(modelPivot.children.length > 0){ 
        modelPivot.remove(modelPivot.children[0]); 
    }

    loader.load(url, (gltf) => {
        const mesh = gltf.scene;
        
        // Normalize Scale (just in case server didn't perfectly)
        const box = new THREE.Box3().setFromObject(mesh);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 1.5 / maxDim; // Fit nicely in view
        
        mesh.position.sub(center); // Center it
        mesh.scale.multiplyScalar(scale);

        modelPivot.rotation.set(0, 0, 0);
        initialModelRotation = modelPivot.rotation.clone();

        modelPivot.add(mesh);
        log("Model loaded into viewport.", "success");
    }, undefined, (error) => {
        console.error(error);
        log("Failed to load 3D model into viewport.", "error");
    });
}

// --- System Diagnostics ---
function startGPUMonitoring() {
    setInterval(async () => {
        try {
            const res = await fetch('/health');
            const data = await res.json();
            
            if (data.status === 'ok') {
                updateGPUUI(data.gpus);
            }
        } catch (e) {
            // log("Failed to fetch GPU stats", "error");
        }
    }, 5000);
}

function updateGPUUI(gpus) {
    if (gpus[0]) {
        gpu0Val.textContent = Math.round(gpus[0].used_vram_mb);
        const p0 = (gpus[0].used_vram_mb / gpus[0].total_vram_mb) * 100;
        gpu0Bar.style.width = `${p0}%`;
        gpu0Bar.style.backgroundColor = p0 > 90 ? '#f00' : p0 > 75 ? '#fa0' : '#4caf50';
    }
    if (gpus[1]) {
        gpu1Val.textContent = Math.round(gpus[1].used_vram_mb);
        const p1 = (gpus[1].used_vram_mb / gpus[1].total_vram_mb) * 100;
        gpu1Bar.style.width = `${p1}%`;
        gpu1Bar.style.backgroundColor = p1 > 90 ? '#f00' : p1 > 75 ? '#fa0' : '#4caf50';
    }
}

// --- Utils ---
function log(msg, type="info") {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = `> ${msg}`;
    consoleOutput.appendChild(entry);
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}
