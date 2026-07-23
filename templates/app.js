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
                    <div class="history-dot ${item.is_delayed ? 'delayed' : 'ontime'}"></div>
                    <div>
                        <p class="history-item-title">${formattedCat}</p>
                        <p class="history-item-sub">$${parseFloat(item.price).toFixed(2)} + $${parseFloat(item.freight).toFixed(2)} freight</p>
                    </div>
                </div>
                <div class="history-item-right">
                    <p class="history-item-status ${item.is_delayed ? 'delayed' : 'ontime'}">
                        ${item.predicted_days.toFixed(1)} Days
                    </p>
                    <p class="history-item-conf">SLA: ${item.estimated_days.toFixed(0)}d</p>
                </div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', itemHtml);
    });
};

const formatDaysToReadable = (daysVal, short = false) => {
    if (daysVal <= 0) {
        return short ? "0h" : "0 hours";
    }
    const days = Math.floor(daysVal);
    const fractionalDays = daysVal - days;
    const hours = Math.round(fractionalDays * 24);
    
    let result = [];
    if (days > 0) {
        result.push(short ? `${days}d` : `${days} day${days === 1 ? '' : 's'}`);
    }
    if (hours > 0) {
        result.push(short ? `${hours}h` : `${hours} hour${hours === 1 ? '' : 's'}`);
    }
    
    return result.length > 0 ? result.join(' ') : (short ? "0h" : "0 hours");
};

// Gauge SVG arc offset visual updates
const updateGaugeOffset = (predictedDays, estimatedDays) => {
    const gaugeFill = document.getElementById('gaugeFill');
    const gaugeText = document.getElementById('gaugeProbText');
    const gaugeSubtext = document.querySelector('.gauge-center-text p');
    
    const delay = Math.max(0, predictedDays - estimatedDays);
    
    // We scale the gauge to a max of 30 days or estimatedDays * 1.5, whichever is higher
    const maxDays = Math.max(30, estimatedDays * 1.5);
    const fraction = Math.min(predictedDays / maxDays, 1.0);
    
    // Half circle SVG stroke dash length is ~125.6
    const strokeDash = 125.6;
    const offset = strokeDash - (strokeDash * fraction);
    
    // Animate fill
    gaugeFill.style.strokeDashoffset = offset;
    
    if (delay > 0) {
        gaugeText.textContent = `+${formatDaysToReadable(delay, true)}`;
        gaugeSubtext.textContent = "Delay Time";
    } else {
        gaugeText.textContent = "0 Days";
        gaugeSubtext.textContent = "Delay Time";
    }
}

// Generate Model Factor Importance list (Explainable AI simulator)
const renderExplainableFactors = (payload, predictedDays, isDelayed) => {
    const list = document.getElementById('factorsList');
    list.innerHTML = '';

    // Calculate contribution factors based on values
    const factors = [
        {
            name: "SLA Window Length",
            impact: payload.estimated_delivery_time_days < 8 ? 2.5 : -1.2,
            desc: payload.estimated_delivery_time_days < 8 ? "Tighter delivery timeframe creates severe logistics pressure" : "Generous SLA window reduces late delivery probability"
        },
        {
            name: "Geospatial Boundary",
            impact: payload.is_same_state === 0 ? 3.8 : -2.1,
            desc: payload.is_same_state === 0 ? "Interstate crossing adds transfer hubs & carrier switches" : "Intra-state routing bypasses national sorting hubs"
        },
        {
            name: "Volumetric Cargo Index",
            impact: (payload.product_weight_g * payload.product_volume_cm3) / 1000000 > 15 ? 1.5 : -0.5,
            desc: (payload.product_weight_g * payload.product_volume_cm3) / 1000000 > 15 ? "Heavy/bulky items limit carrier dispatch options" : "Standard size cargo matches normal courier streams"
        },
        {
            name: "Financial Premium",
            impact: payload.price > 300 ? 0.8 : -0.4,
            desc: payload.price > 300 ? "High value items require additional secure handling processes" : "Standard item pricing facilitates rapid processing"
        }
    ];

    // Sort factors by absolute impact
    factors.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));

    factors.forEach(f => {
        const percent = Math.min(Math.abs(f.impact) * 20, 100); // Scale factor for presentation
        const isPos = f.impact > 0;
        
        const itemHtml = `
            <div class="factor-item relative group cursor-help">
                <span class="factor-name">${f.name}</span>
                <div class="factor-bar-wrapper">
                    <div class="factor-bar-bg">
                        <div class="factor-bar-fill ${isPos ? 'pos' : 'neg'}" style="width: ${percent}%"></div>
                    </div>
                    <span class="factor-value ${isPos ? 'pos' : 'neg'}">${isPos ? '+' : ''}${f.impact.toFixed(1)}d</span>
                </div>
                <!-- Dynamic XAI Tooltip -->
                <span class="tooltip-text" style="width: 250px; bottom: 130%; font-size: 0.65rem;">
                    ${f.desc} (${isPos ? 'Adds' : 'Subtracts'} ~${Math.abs(f.impact.toFixed(1))} days)
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
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) throw new Error("API Offline");
        resultData = await response.json();
    } catch (err) {
        console.warn("Backend FastAPI offline. Using sandbox fallback logic...");
        // Fallback simulation logic matching ML profiles
        const baseDays = 8.5 + (payload.freight_value > 30 ? 4.0 : 0) + (payload.is_same_state === 0 ? 8.0 : 0) + (payload.product_weight_g > 5000 ? 3.0 : 0);
        resultData = {
            predicted_days: baseDays
        };
    }

    // Delay result render briefly for loading animation feel
    setTimeout(() => {
        loadingState.classList.add('hidden');
        resultState.className = "state-panel";
        resultMeta.className = "result-meta-row";
        
        // Reset submit button state
        submitBtn.disabled = false;
        submitBtn.innerHTML = `<i data-lucide="sparkles" style="width:1.1rem; height:1.1rem"></i><span>Run Predictor</span>`;
        lucide.createIcons();

        const predictedDays = resultData.predicted_days;
        const estimatedDays = payload.estimated_delivery_time_days;
        const isDelayed = predictedDays > estimatedDays;
        const delayDays = Math.max(0, predictedDays - estimatedDays);

        // Render outcomes
        const titleEl = document.getElementById('outcomeLabel');
        const descEl = document.getElementById('outcomeDesc');

        if (isDelayed) {
            badge.className = "badge-onnx delayed";
            badge.textContent = "Delayed";
            titleEl.className = "outcome-title delayed";
            titleEl.textContent = `Predicted: ${formatDaysToReadable(predictedDays)}`;
            descEl.textContent = `The package you ordered will be delivered in ${formatDaysToReadable(predictedDays)}, and the model predicts a delivery delay of ${formatDaysToReadable(delayDays)}.`;
        } else {
            badge.className = "badge-onnx ontime";
            badge.textContent = "On-Time";
            titleEl.className = "outcome-title ontime";
            titleEl.textContent = `Predicted: ${formatDaysToReadable(predictedDays)}`;
            descEl.textContent = `The package you ordered will be delivered in ${formatDaysToReadable(predictedDays)}, and the model predicts a delivery delay of 0 hours (on time).`;
        }

        updateGaugeOffset(predictedDays, estimatedDays);
        renderExplainableFactors(payload, predictedDays, isDelayed);

        // Metadata rendering
        document.getElementById('metaRoute').textContent = payload.is_same_state === 1 ? "Simple Domestic" : "Interstate Transit";
        const metricIndex = (payload.product_weight_g * payload.product_volume_cm3) / 1000000;
        document.getElementById('metaWeight').textContent = metricIndex.toFixed(2) + " Index";

        // Save
        saveToHistoryLog({
            category: payload.product_category_name,
            price: payload.price,
            freight: payload.freight_value,
            predicted_days: predictedDays,
            estimated_days: estimatedDays,
            is_delayed: isDelayed
        });

    }, 850);
});

// Run Init Actions
populateFormSelects();
setupSliderBindings();
loadHistoryLog();
// Load standard preset as default start values
loadPreset('standard');
