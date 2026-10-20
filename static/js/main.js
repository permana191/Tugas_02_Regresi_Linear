document.addEventListener('DOMContentLoaded', function () {
    const likesInput = document.getElementById('likes_given');
    const likesSlider = document.getElementById('likes_slider');
    const liveEstimate = document.getElementById('liveEstimate');
    const liveRatio = document.getElementById('liveRatio');

    // Nilai default intersep dan slope dari model
    const INTERCEPT = 25.12;
    const SLOPE = 0.32;

    function updateLiveCalculation(val) {
        const x = parseFloat(val) || 0;
        const yPred = Math.max(0, Math.round(INTERCEPT + (SLOPE * x)));
        const ratio = x > 0 ? ((yPred / x) * 100).toFixed(1) : 0;

        if (liveEstimate) liveEstimate.textContent = yPred.toLocaleString() + ' Likes';
        if (liveRatio) liveRatio.textContent = ratio + '% Ratio';
    }

    // Sinkronisasi Input Angka dan Range Slider
    if (likesInput && likesSlider) {
        likesInput.addEventListener('input', function () {
            likesSlider.value = this.value;
            updateLiveCalculation(this.value);
        });

        likesSlider.addEventListener('input', function () {
            likesInput.value = this.value;
            updateLiveCalculation(this.value);
        });

        // Inisialisasi awal
        updateLiveCalculation(likesInput.value);
    }
});