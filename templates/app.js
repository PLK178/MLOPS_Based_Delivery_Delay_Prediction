// Initialize Lucide Icons
lucide.createIcons();

// Dropdowns Options population
const populateFormSelects = () => {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const weekdays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    const curDate = new Date();

    const monthSelect = document.getElementById('purchase_month');
    months.forEach((m, idx) => {
        const option = new Option(m, idx + 1, false, idx === curDate.getMonth());
        monthSelect.add(option);
    });

    const weekdaySelect = document.getElementById('purchase_day_of_week');
    weekdays.forEach((w, idx) => {
        const option = new Option(w, idx, false, idx === curDate.getDay());
        weekdaySelect.add(option);
    });

    const hourSelect = document.getElementById('purchase_hour');
    for (let h = 0; h < 24; h++) {
        const hourLabel = h.toString().padStart(2, '0') + ':00';
        const option = new Option(hourLabel, h, false, h === 14);
        hourSelect.add(option);
    }
};

// Slider value sync bindings
const setupSliderBindings = () => {
    const sliderBindings = [
        { id: 'product_weight_g', displayId: 'weightVal', format: (val) => val >= 1000 ? `${(val/1000).toFixed(1)} kg` : `${val} g` },
        { id: 'product_volume_cm3', displayId: 'volumeVal', format: (val) => `${parseInt(val).toLocaleString()} cm³` },
        { id: 'estimated_delivery_time_days', displayId: 'deliveryVal', format: (val) => `${val} Days` }
    ];

    sliderBindings.forEach(binding => {
        const el = document.getElementById(binding.id);
        const display = document.getElementById(binding.displayId);
        
        el.addEventListener('input', () => {
            display.textContent = binding.format(el.value);
        });
    });
};

// Preset Scenarios Data Loading
const loadPreset = (type) => {
    const presets = {
        standard: {
            category: "beleza_saude",
            sameState: true,
            price: 79.90,
            freight: 12.40,
            weight: 350,
            volume: 1800,
            delivery: 15.0
        },
        heavy_delayed: {
            category: "utilidades_domesticas",
            sameState: false,
            price: 289.00,
            freight: 65.50,
            weight: 9500,
            volume: 24000,
            delivery: 5.5 // very short estimated SLA delivery window
        },
        express: {
            category: "informatica_acessorios",
            sameState: true,
            price: 450.00,
            freight: 8.50,
            weight: 600,
            volume: 3200,
            delivery: 24.0 // generous SLA window
        }
    };

    const data = presets[type];
    if (!data) return;

    // Apply values to DOM
    document.getElementById('product_category_name').value = data.category;
    document.getElementById('is_same_state').checked = data.sameState;
    document.getElementById('price').value = data.price.toFixed(2);
    document.getElementById('freight_value').value = data.freight.toFixed(2);
    
    document.getElementById('product_weight_g').value = data.weight;
    document.getElementById('product_weight_g').dispatchEvent(new Event('input'));

    document.getElementById('product_volume_cm3').value = data.volume;
    document.getElementById('product_volume_cm3').dispatchEvent(new Event('input'));

    document.getElementById('estimated_delivery_time_days').value = data.delivery;
    document.getElementById('estimated_delivery_time_days').dispatchEvent(new Event('input'));
};

// Local storage registry for runs history log
let runHistory = [];
const loadHistoryLog = () => {
    const saved = localStorage.getItem('vanilla_runs_history');
    if (saved) {
        try {
            runHistory = JSON.parse(saved);
            renderHistoryLog();
        } catch (e) {
            console.error(e);
        }
    }
};

const saveToHistoryLog = (item) => {
    runHistory.unshift(item);
    runHistory = runHistory.slice(0, 10);
    localStorage.setItem('vanilla_runs_history', JSON.stringify(runHistory));
    renderHistoryLog();
};

const renderHistoryLog = () => {
    const card = document.getElementById('historyCard');
    const container = document.getElementById('historyList');
    
    if (runHistory.length === 0) {
        card.classList.add('hidden');
        return;
    }
    
    card.classList.remove('hidden');
    container.innerHTML = '';

    runHistory.forEach(item => {
        const formattedCat = item.category.replace(/_/g, ' ');
        const itemHtml = `
            <div class="history-item">
                <div class="history-item-left">
                    <div class="history-dot ${item.prediction === 1 ? 'delayed' : 'ontime'}"></div>
                    <div>
                        <p class="history-item-title">${formattedCat}</p>
                        <p class="history-item-sub">$${parseFloat(item.price).toFixed(2)} + $${parseFloat(item.freight).toFixed(2)} freight</p>
                    </div>
                </div>
                <div class="history-item-right">
                    <p class="history-item-status ${item.prediction === 1 ? 'delayed' : 'ontime'}">
                        ${item.prediction === 1 ? 'Delayed' : 'On-Time'}
                    </p>
                    <p class="history-item-conf">${(item.probability * 100).toFixed(0)}% conf.</p>
                </div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', itemHtml);
    });
};

// Gauge SVG arc offset visual updates
const updateGaugeOffset = (prob) => {
    const gaugeFill = document.getElementById('gaugeFill');
    const gaugeText = document.getElementById('gaugeProbText');
    
    // Half circle SVG stroke dash length is ~125.6
    const strokeDash = 125.6;
    const offset = strokeDash - (strokeDash * prob);
    
    // Animate fill
    gaugeFill.style.strokeDashoffset = offset;
    gaugeText.textContent = (prob * 100).toFixed(0) + "%";

    // Dynamic color gradient shifting
    if (prob < 0.35) {
        gaugeFill.className.baseVal = "gauge-fill ontime";
    } else if (prob < 0.65) {
        gaugeFill.className.baseVal = "gauge-fill warning";
    } else {
        gaugeFill.className.baseVal = "gauge-fill critical";
    }
};

// Generate Model Factor Importance list (Explainable AI simulator)
const renderExplainableFactors = (payload, prob, isDelayed) => {
    const list = document.getElementById('factorsList');
    list.innerHTML = '';

    // Calculate contribution factors based on values
    const factors = [
        {
            name: "SLA Window Length",
            impact: payload.estimated_delivery_time_days < 8 ? 0.35 : -0.15,
            desc: payload.estimated_delivery_time_days < 8 ? "Tighter delivery timeframe creates severe logistics pressure" : "Generous SLA window reduces late delivery probability"
        },
        {
            name: "Geospatial Boundary",
            impact: payload.is_same_state === 0 ? 0.18 : -0.12,
            desc: payload.is_same_state === 0 ? "Interstate crossing adds transfer hubs & carrier switches" : "Intra-state routing bypasses national sorting hubs"
        },
        {
            name: "Volumetric Cargo Index",
            impact: (payload.product_weight_g * payload.product_volume_cm3) / 1000000 > 15 ? 0.22 : -0.05,
            desc: (payload.product_weight_g * payload.product_volume_cm3) / 1000000 > 15 ? "Heavy/bulky items limit carrier dispatch options" : "Standard size cargo matches normal courier streams"
        },
        {
            name: "Financial Premium",
            impact: payload.price > 300 ? 0.1 : -0.05,
            desc: payload.price > 300 ? "High value items require additional secure handling processes" : "Standard item pricing facilitates rapid processing"
        }
    ];

    // Sort factors by absolute impact
    factors.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));

    factors.forEach(f => {
        const percent = Math.min(Math.abs(f.impact) * 200, 100); // Scale factor for presentation
        const isPos = f.impact > 0;
        
        const itemHtml = `
            <div class="factor-item relative group cursor-help">
                <span class="factor-name">${f.name}</span>
                <div class="factor-bar-wrapper">
                    <div class="factor-bar-bg">
                        <div class="factor-bar-fill ${isPos ? 'pos' : 'neg'}" style="width: ${percent}%"></div>
                    </div>
                    <span class="factor-value ${isPos ? 'pos' : 'neg'}">${isPos ? '+' : ''}${Math.round(f.impact * 100)}%</span>
                </div>
                <!-- Dynamic XAI Tooltip -->
                <span class="tooltip-text" style="width: 250px; bottom: 130%; font-size: 0.65rem;">
                    ${f.desc} (${isPos ? 'Increases' : 'Decreases'} risk by ${Math.abs(Math.round(f.impact * 100))}%)
                </span>
            </div>
        `;
        list.insertAdjacentHTML('beforeend', itemHtml);
    });
};

// Form submission handler
const form = document.getElementById('predictionForm');
form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById('submitBtn');
    const emptyState = document.getElementById('emptyState');
    const loadingState = document.getElementById('loadingState');
    const resultState = document.getElementById('resultState');
    const resultMeta = document.getElementById('resultMeta');
    const badge = document.getElementById('badge');

    // Toggle loading states
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<div class="spinner"></div><span>Running...</span>`;
    emptyState.classList.add('hidden');
    resultState.classList.add('hidden');
    resultMeta.classList.add('hidden');
    loadingState.classList.remove('hidden');
    badge.className = "badge-onnx";

    // Retrieve input values
    const payload = {
        price: parseFloat(document.getElementById('price').value),
        freight_value: parseFloat(document.getElementById('freight_value').value),
        product_category_name: document.getElementById('product_category_name').value,
        product_weight_g: parseInt(document.getElementById('product_weight_g').value),
        product_volume_cm3: parseInt(document.getElementById('product_volume_cm3').value),
        is_same_state: document.getElementById('is_same_state').checked ? 1 : 0,
        purchase_month: parseInt(document.getElementById('purchase_month').value),
        purchase_day_of_week: parseInt(document.getElementById('purchase_day_of_week').value),
        purchase_hour: parseInt(document.getElementById('purchase_hour').value),
        estimated_delivery_time_days: parseFloat(document.getElementById('estimated_delivery_time_days').value)
    };

    let resultData;
    try {
        const response = await fetch('https://mlops-based-delivery-delay-prediction.onrender.com/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) throw new Error("API Offline");
        resultData = await response.json();
    } catch (err) {
        console.warn("Backend FastAPI offline. Using sandbox fallback logic...");
        // Fallback simulation logic matching ML profiles
        const risk = (
            (payload.freight_value > 30 ? 0.30 : 0.05) +
            (payload.estimated_delivery_time_days < 7 ? 0.45 : 0.05) +
            (payload.product_weight_g > 5000 ? 0.22 : 0.0) +
            (payload.is_same_state === 0 ? 0.18 : 0.0)
        );
        resultData = {
            prediction: risk > 0.45 ? 1 : 0,
            probability: Math.min(Math.max(risk, 0.05), 0.95)
        };
    }

    // Delay result render briefly for loading animation feel
    setTimeout(() => {
        loadingState.classList.add('hidden');
        resultState.classList.remove('hidden');
        resultMeta.classList.remove('hidden');
        
        // Reset submit button state
        submitBtn.disabled = false;
        submitBtn.innerHTML = `<i data-lucide="sparkles" style="width:1.1rem; height:1.1rem"></i><span>Run ONNX Predictor</span>`;
        lucide.createIcons();

        const isDelayed = resultData.prediction === 1;
        const confidence = resultData.probability;

        // Render outcomes
        const titleEl = document.getElementById('outcomeLabel');
        const descEl = document.getElementById('outcomeDesc');

        if (isDelayed) {
            badge.className = "badge-onnx delayed";
            badge.textContent = "High Risk";
            titleEl.className = "outcome-title delayed";
            titleEl.textContent = "High Delay Risk";
            descEl.textContent = "This transit profile exceeds safe limits for standard operational SLA guarantees.";
        } else {
            badge.className = "badge-onnx ontime";
            badge.textContent = "Safe Profile";
            titleEl.className = "outcome-title ontime";
            titleEl.textContent = "Likely On-Time";
            descEl.textContent = "Standard route features are optimized. The shipment is projected to land within SLA.";
        }

        updateGaugeOffset(confidence);
        renderExplainableFactors(payload, confidence, isDelayed);

        // Metadata rendering
        document.getElementById('metaRoute').textContent = payload.is_same_state === 1 ? "Simple Domestic" : "Interstate Transit";
        const metricIndex = (payload.product_weight_g * payload.product_volume_cm3) / 1000000;
        document.getElementById('metaWeight').textContent = metricIndex.toFixed(2) + " Index";

        // Save
        saveToHistoryLog({
            category: payload.product_category_name,
            price: payload.price,
            freight: payload.freight_value,
            prediction: resultData.prediction,
            probability: confidence
        });

    }, 850);
});

// Run Init Actions
populateFormSelects();
setupSliderBindings();
loadHistoryLog();
// Load standard preset as default start values
loadPreset('standard');
