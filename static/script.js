// Global variables
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const predictionDiv = document.getElementById('prediction');
const predictionContainer = document.getElementById('prediction-container');
const mainContainer = document.getElementById('main-container');

const CATEGORIES = [
    "airplane", "apple", "banana", "bed", "bicycle", "bird", "cake", "car", "cat", "chair",
    "clock", "dog", "fish", "flower", "frog", "guitar", "hat", "house", "tree", "sun"
];

ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;
canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    draw(e);
});
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', () => {
    drawing = false;
    ctx.beginPath();
});
canvas.addEventListener('mouseleave', () => {
    drawing = false;
    ctx.beginPath();
});

function draw(e) {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    ctx.lineWidth = 12;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

clearBtn.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    predictionDiv.innerHTML = '';
    predictionContainer.style.display = 'none';
    mainContainer.classList.remove("expanded");
});

function getCanvasImageBase64() {
    return canvas.toDataURL('image/png');
}

predictBtn.addEventListener('click', async () => {
    const imageData = getCanvasImageBase64();

    try {
        predictionDiv.innerHTML = "Predicting...";
        predictionDiv.style.fontFamily = "ArcadeClassic, Arial, sans-serif";
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({image: imageData})
        });
        const result = await response.json();

        if (result.error) {
            predictionDiv.innerHTML = "Error: " + result.error;
        } else {
            mainContainer.classList.add("expanded");
            predictionContainer.style.display = 'flex';

            let positivePredictions = result.predictions
                .map((prob, index) => ({name: CATEGORIES[index], prob}))
                .filter(entry => entry.prob > 0);
            let zeroPredictions = result.predictions
                .map((prob, index) => ({name: CATEGORIES[index], prob}))
                .filter(entry => entry.prob === 0);

            zeroPredictions.sort(() => 0.5 - Math.random());
            let selectedZeroPredictions = zeroPredictions.slice(0, 3);

            let finalPredictions = [...positivePredictions, ...selectedZeroPredictions];

            finalPredictions.sort((a, b) => b.prob - a.prob);

            let topPrediction = finalPredictions[0];

            predictionDiv.innerHTML = finalPredictions.map((entry, index) =>
                `<strong style="color:${index === 0 ? "#50FA7B" : "white"}">${entry.name.toUpperCase()}</strong>
                 <span style="float:right; color:${index === 0 ? "#50FA7B" : "white"};">${(entry.prob * 100).toFixed(2)}%</span>`
            ).join("<br>");
        }
    } catch (error) {
        predictionDiv.innerHTML = "Error during prediction: " + error;
        console.error(error);
    }
});
