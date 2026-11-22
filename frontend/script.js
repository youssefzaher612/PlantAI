
// =====================================
// GET HTML ELEMENTS
// =====================================

const imageInput = document.getElementById("imageInput");
const uploadArea = document.getElementById("uploadArea");
const previewContainer = document.getElementById("previewContainer");
const imagePreview = document.getElementById("imagePreview");
const analyzeButton = document.getElementById("analyzeButton");
const removeImage = document.getElementById("removeImage");
const statusMessage = document.getElementById("statusMessage");

const resultCard = document.getElementById("resultCard");
const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");
const secondPrediction = document.getElementById("secondPrediction");
const warningMessage = document.getElementById("warningMessage");
const resetButton = document.getElementById("resetButton");

const totalLeaves = document.getElementById("totalLeaves");
const segmentationSection = document.getElementById("segmentationSection");
const segmentedImage = document.getElementById("segmentedImage");
const leafResultsSection = document.getElementById("leafResultsSection");
const leafResults = document.getElementById("leafResults");

let selectedFile = null;


// =====================================
// FORMAT CLASS NAMES
// =====================================

function formatClass(className) {

    if (!className) {
        return "Unknown";
    }

    const names = {
        "faba_blight": "Blight",
        "faba_chocolate": "Chocolate Spot",
        "faba_healthy": "Healthy",
        "faba_rust": "Rust"
    };

    return names[className] ||
        className
            .replace("faba_", "")
            .replaceAll("_", " ")
            .replace(/\b\w/g, letter => letter.toUpperCase());
}


// =====================================
// FORMAT CONFIDENCE
// Backend returns decimal: 0.68
// Display: 68.00%
// =====================================

function formatConfidence(value) {

    if (value === null || value === undefined) {
        return "N/A";
    }

    return (Number(value) * 100).toFixed(2) + "%";
}


// =====================================
// STATUS MESSAGE
// =====================================

function showStatus(message, type = "") {

    statusMessage.textContent = message;
    statusMessage.className = "status-message";

    if (type) {
        statusMessage.classList.add(type);
    }
}


// =====================================
// SELECT IMAGE
// =====================================

imageInput.addEventListener("change", () => {

    const file = imageInput.files[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {

        showStatus("Please select a valid image.", "error");
        return;
    }

    selectedFile = file;

    imagePreview.src = URL.createObjectURL(file);

    previewContainer.hidden = false;
    uploadArea.hidden = true;

    analyzeButton.disabled = false;

    resultCard.hidden = true;

    showStatus("");
});


// =====================================
// REMOVE IMAGE
// =====================================

removeImage.addEventListener("click", clearImage);


function clearImage() {

    selectedFile = null;
    imageInput.value = "";

    imagePreview.src = "";
    segmentedImage.src = "";

    previewContainer.hidden = true;
    uploadArea.hidden = false;

    analyzeButton.disabled = true;

    resultCard.hidden = true;
    segmentationSection.hidden = true;
    leafResultsSection.hidden = true;
    warningMessage.hidden = true;

    leafResults.innerHTML = "";
    totalLeaves.textContent = "";

    prediction.textContent = "";
    confidence.textContent = "";
    secondPrediction.textContent = "";

    showStatus("");
}


// =====================================
// ANALYZE IMAGE
// =====================================

analyzeButton.addEventListener("click", async () => {

    if (!selectedFile) return;

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing...";

    resultCard.hidden = true;

    segmentationSection.hidden = true;
    leafResultsSection.hidden = true;

    showStatus(
        "SAM is detecting leaf regions and AI is analyzing your image. Please wait..."
    );

    const formData = new FormData();

    formData.append("file", selectedFile);

    try {

        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {

            throw new Error(
                data.detail || "Prediction failed."
            );
        }

        if (!data.success) {

            throw new Error(
                data.message || "No leaf regions were detected."
            );
        }

        if (!data.results || data.results.length === 0) {

            throw new Error(
                "No leaf diagnosis results were returned."
            );
        }

        displayResults(data);

        resultCard.hidden = false;

        showStatus(
            "Analysis completed successfully.",
            "success"
        );

        resultCard.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    } catch (error) {

        console.error("Prediction error:", error);

        showStatus(
            "Error: " + error.message,
            "error"
        );

    } finally {

        analyzeButton.disabled = false;
        analyzeButton.textContent = "Analyze Image";
    }

});


// =====================================
// DISPLAY MAIN DIAGNOSIS
// =====================================

function displayResults(data) {

    const firstResult = data.results[0];

    // Number of detected leaves
    totalLeaves.textContent =
        `SAM detected ${data.total_leaves ?? data.results.length} leaf region(s).`;


    // Main prediction
    prediction.textContent =
        formatClass(firstResult.prediction);


    // Main confidence
    confidence.textContent =
        `Confidence: ${formatConfidence(firstResult.confidence)}`;


    // Second possible class
    secondPrediction.textContent =
        firstResult.second_prediction
            ? `${formatClass(firstResult.second_prediction)} (${formatConfidence(firstResult.second_confidence)})`
            : "Not available";


    // Uncertainty warning
    warningMessage.hidden = !firstResult.uncertain;


    // SAM segmentation image
    displaySegmentation(data.segmented_image);


    // Individual leaf diagnoses
    displayIndividualLeaves(data.results);
}


// =====================================
// DISPLAY SAM SEGMENTATION IMAGE
// =====================================

function displaySegmentation(base64Image) {

    segmentedImage.onload = () => {

        segmentationSection.hidden = false;

        console.log("SAM segmentation image loaded successfully.");
    };

    segmentedImage.onerror = () => {

        segmentationSection.hidden = true;

        console.error("Could not load SAM segmentation image.");

        showStatus(
            "Diagnosis completed, but the segmentation image could not be displayed.",
            "error"
        );
    };

    if (!base64Image) {

        segmentationSection.hidden = true;

        console.error("Backend did not return segmented_image.");

        return;
    }

    // Remove old image before displaying new one
    segmentedImage.removeAttribute("src");

    segmentedImage.src =
        `data:image/png;base64,${base64Image}`;
}


// =====================================
// DISPLAY INDIVIDUAL LEAF RESULTS
// =====================================

function displayIndividualLeaves(results) {

    leafResults.innerHTML = "";

    if (!results || results.length === 0) {

        leafResultsSection.hidden = true;
        return;
    }

    results.forEach(leaf => {

        const card = document.createElement("div");

        card.className = "leaf-card";


        // Extracted leaf image
        if (leaf.crop_image) {

            const image = document.createElement("img");

            image.src =
                `data:image/png;base64,${leaf.crop_image}`;

            image.alt = `Leaf ${leaf.leaf_number}`;

            card.appendChild(image);
        }


        // Leaf number
        const title = document.createElement("h4");

        title.textContent =
            `Leaf ${leaf.leaf_number}`;

        card.appendChild(title);


        // Prediction
        const diagnosis = document.createElement("p");

        diagnosis.textContent =
            `Diagnosis: ${formatClass(leaf.prediction)}`;

        card.appendChild(diagnosis);


        // Confidence
        const confidenceText = document.createElement("p");

        confidenceText.className = "leaf-confidence";

        confidenceText.textContent =
            `Confidence: ${formatConfidence(leaf.confidence)}`;

        card.appendChild(confidenceText);


        // Second prediction
        if (leaf.second_prediction) {

            const alternative = document.createElement("p");

            alternative.textContent =
                `Alternative: ${formatClass(leaf.second_prediction)} (${formatConfidence(leaf.second_confidence)})`;

            card.appendChild(alternative);
        }


        // Uncertainty warning
        if (leaf.uncertain) {

            const warning = document.createElement("p");

            warning.className = "leaf-warning";

            warning.textContent =
                "⚠ Uncertain prediction";

            card.appendChild(warning);
        }


        leafResults.appendChild(card);

    });

    leafResultsSection.hidden = false;
}


// =====================================
// ANALYZE ANOTHER IMAGE
// =====================================

resetButton.addEventListener("click", () => {

    clearImage();

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });

});


// =====================================
// DRAG AND DROP
// =====================================

uploadArea.addEventListener("dragover", event => {

    event.preventDefault();

    uploadArea.style.borderColor = "#347b45";

});


uploadArea.addEventListener("dragleave", () => {

    uploadArea.style.borderColor = "#a8c9ad";

});


uploadArea.addEventListener("drop", event => {

    event.preventDefault();

    const file = event.dataTransfer.files[0];

    if (!file || !file.type.startsWith("image/")) {

        showStatus(
            "Please drop a valid image.",
            "error"
        );

        return;
    }

    imageInput.files = event.dataTransfer.files;

    imageInput.dispatchEvent(
        new Event("change")
    );

});
