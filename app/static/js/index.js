// Analysis Site
(() => {
  let timerInterval = null;
  let seconds = 0;


  // Show loading spinner and timer
  const showSpinner = () => {
    document.getElementById('loading-spinner').style.display = 'block';
    seconds = 0;
    document.getElementById('timer-counter').textContent = seconds;
    timerInterval = setInterval(() => {
      seconds++;
      document.getElementById('timer-counter').textContent = seconds;
    }, 1000);
  };


  // Hide spinner and stop timer
  const hideSpinner = () => {
    clearInterval(timerInterval);
    document.getElementById('timer-counter').textContent = `Analyzed in ${seconds} seconds`;
    document.getElementById('loading-spinner').style.display = 'none';
  };


  // Clear previous results and states
  const clearResults = () => {
    document.getElementById('prediction-result').innerHTML = "";
    document.getElementById('lime-visual').innerHTML = "";
    document.getElementById('results-placeholder').style.display = 'none';
    removeErrorClasses();
  };


  // Remove error 
  const removeErrorClasses = () => {
    const subjectField = document.getElementById('subject');
    const bodyField = document.getElementById('body');
    subjectField.classList.remove('error', 'analyzing');
    bodyField.classList.remove('error', 'analyzing');
  };


  // Auto resize textarea element
  const autoResizeTextarea = (element) => {
    element.style.height = 'auto';
    element.style.height = `${element.scrollHeight}px`;
  };


  // Handle analysis of input
  const analyzeEmail = () => {
    clearResults();

    const subjectField = document.getElementById('subject');
    const bodyField = document.getElementById('body');
    const subject = subjectField.value.trim();
    const body = bodyField.value.trim();

    if (!subject && !body) {
      subjectField.classList.add('error');
      bodyField.classList.add('error');
      alert("Please provide at least an Email Subject or Email Body.");
      return;
    }

    // "analyzing" visual state
    subjectField.classList.add('analyzing');
    bodyField.classList.add('analyzing');

    showSpinner();

    fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subject, body })
    })
      .then(response => response.json())
      .then(data => {
        hideSpinner();
        displayResults(data);
      })
      .catch(error => {
        hideSpinner();
        console.error('Error analyzing email:', error);
      });
  };


  // Display analysis results
  const displayResults = (data) => {
    const predictionResult = document.getElementById('prediction-result');
    predictionResult.innerHTML = `
      <p><strong>Prediction:</strong> ${data.predicted_label}</p>
      <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2) + "%"}</p>
    `;

    // LIME iframe
    const limeIframe = document.createElement('iframe');
    limeIframe.style.width = "100%";
    limeIframe.style.height = "450px";
    limeIframe.style.border = "none";
    limeIframe.srcdoc = data.lime_html;
    const limeVisual = document.getElementById('lime-visual');
    limeVisual.innerHTML = "";
    limeVisual.appendChild(limeIframe);

    // Integrated Gradients 
    document.getElementById('ig-visual').innerHTML = data.ig_html;
    document.getElementById('results-placeholder').style.display = 'block';
  };


  // Init event listeners
  const initEventListeners = () => {
    document.getElementById('analyze-btn').addEventListener('click', analyzeEmail);
    
    document.getElementById('subject').addEventListener('focus', removeErrorClasses);
    document.getElementById('body').addEventListener('focus', removeErrorClasses);
    
    const bodyField = document.getElementById('body');
    bodyField.addEventListener('input', () => autoResizeTextarea(bodyField));
    document.addEventListener("DOMContentLoaded", () => autoResizeTextarea(bodyField));
  };


  const init = () => {
    initEventListeners();
  };

  
  init();
})();
