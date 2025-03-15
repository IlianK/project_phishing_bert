// Learn Phishing Site
(() => {
    let emailIndex = 0;
    const totalEmails = 10;
    let currentLanguage = 'en';
  
    // Fetch and display (by index)
    const fetchEmail = (index) => {
      clearEmailDisplay();
      disableButton('check-email');
  
      fetch('/get_email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email_index: index })
      })
        .then(response => response.json())
        .then(data => {
          if (data.error) return console.error(data.error);
  
          document.querySelector('#email-subject span').innerHTML = tokenize(data.subject);
          document.getElementById('email-body').innerHTML = tokenize(data.body);
          resetSelections();
          updateNavigationButtons();
          updateCheckButtonState();
        })
        .catch(error => console.error('Error fetching email:', error));
    };
  
    // Tokenize text into clickable parts
    const tokenize = (text) => {
      return text
        .split(' ')
        .map(word => `<span class="token">${word}</span>`)
        .join(' ');
    };
  
    // Init click events email tokens in box
    const initializeEmailBoxClick = () => {
      const emailBox = document.querySelector('.email-box');
      emailBox.addEventListener('click', (event) => {
        if (event.target.classList.contains('token')) {
          const checkbox = document.getElementById('mark-legit');
          if (checkbox.checked) checkbox.checked = false;
          event.target.classList.toggle('highlighted');
          updateCheckButtonState();
        }
      });
    };
  
    // Update state of Check button
    const updateCheckButtonState = () => {
      const highlightedTokens = document.querySelectorAll('.token.highlighted');
      const checkButton = document.getElementById('check-email');
      const checkbox = document.getElementById('mark-legit');
      checkButton.disabled = !(highlightedTokens.length > 0 || checkbox.checked);
    };
  
    // Update the navigation (prev/next)
    const updateNavigationButtons = () => {
      document.getElementById('prev-email').disabled = (emailIndex === 0);
      document.getElementById('next-email').disabled = (emailIndex >= totalEmails - 1);
    };
  
    // Clear email display and results
    const clearEmailDisplay = () => {
      document.getElementById('lime-visual').innerHTML = "";
      document.getElementById('results-placeholder').style.display = 'none';
      document.getElementById('result-message').style.display = 'none';
      document.getElementById('result-message').textContent = '';
    };
  
    // Reset checkbox and remove highlighted tokens
    const resetSelections = () => {
      document.getElementById('mark-legit').checked = false;
      document.querySelectorAll('.token.highlighted').forEach(token => token.classList.remove('highlighted'));
      document.getElementById('true-label').textContent = "";
      document.getElementById('model-prediction').textContent = "";
      document.getElementById('confidence-score').textContent = "";
      document.getElementById('lime-visual').innerHTML = "";
    };
  
    // Disable button by ID
    const disableButton = (id) => {
      document.getElementById(id).disabled = true;
    };
  
    // Handle click on Check button
    const handleCheckEmail = () => {
        const resultMessage = document.getElementById('result-message');
        const predictionResult = document.getElementById('prediction-result');
        // const highlightedWordsList = document.getElementById('highlighted-words-list');
        const resultsPlaceholder = document.getElementById('results-placeholder');
    
        const highlightedTokens = Array.from(document.querySelectorAll('.token.highlighted'))
                                    .map(token => token.textContent);
    
        fetch('/get_email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email_index: emailIndex })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) return console.error(data.error);
    
            const isPhishing = data.true_label === 1;
            const predictedLabel = data.pred_label;
            const confidenceScore = (data.confidence * 100).toFixed(2) + "%";
    
            resultMessage.style.display = 'block';
            resultMessage.innerHTML = isPhishing
            ? `<p style="color: red; font-weight: bold;">This email is a phishing attempt.</p>`
            : `<p style="color: green; font-weight: bold;">This email is legit.</p>`;
    
            predictionResult.innerHTML =
            `<p><strong>Prediction:</strong> ${predictedLabel}</p>
            <p><strong>Confidence:</strong> ${confidenceScore}</p>`;
    
            fetch('/get_lime_html', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email_index: emailIndex })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) return console.error(data.error);
                const limeVisual = document.getElementById('lime-visual');
                limeVisual.innerHTML = "";
                const iframe = document.createElement('iframe');
                iframe.style.width = "100%";
                iframe.style.height = "600px";
                iframe.style.border = "none";
                iframe.srcdoc = data.lime_html;
                limeVisual.appendChild(iframe);
    
                // highlightedWordsList.innerHTML = highlightedTokens.length > 0
                //   ? `<strong>You Highlighted:</strong> ${highlightedTokens.join(', ')}`
                //   : "";
                
                resultsPlaceholder.style.display = 'block';
            })
            .catch(error => console.error('Error fetching LIME HTML:', error));
        })
        .catch(error => console.error('Error fetching email data:', error));
    };
    
  
    // Toggle info box visibility
    const toggleInfoBox = () => {
      const infoBox = document.getElementById('info-box');
      const emailContent = document.getElementById('email-content');
      const navButtons = document.getElementById('nav-buttons');
      const resultsPlaceholder = document.getElementById('results-placeholder');
  
      if (infoBox.style.display === 'none') {
        infoBox.style.display = 'block';
        emailContent.style.display = 'none';
        navButtons.style.display = 'none';
        resultsPlaceholder.style.display = 'none';
      } else {
        infoBox.style.display = 'none';
        emailContent.style.display = 'block';
        navButtons.style.display = 'flex';
      }
    };
  
    // Switch language and refresh 
    const handleLanguageSwitch = (event) => {
      currentLanguage = event.target.checked ? 'de' : 'en';
      fetch('/set_language', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lang: currentLanguage })
      })
        .then(response => response.json())
        .then(data => {
          console.log(`Language switched to: ${data.lang}`);
          if (currentLanguage === 'de') {
            document.getElementById('instructions-en').style.display = 'none';
            document.getElementById('instructions-de').style.display = 'block';
            document.getElementById('continue-btn').textContent = "Weiter";
          } else {
            document.getElementById('instructions-en').style.display = 'block';
            document.getElementById('instructions-de').style.display = 'none';
            document.getElementById('continue-btn').textContent = "Continue";
          }
          fetchEmail(emailIndex);
        })
        .catch(err => console.error('Error switching language:', err));
    };
  
    // Init event listeners 
    const initEventListeners = () => {
      document.getElementById('mark-legit').addEventListener('change', () => {
        if (document.getElementById('mark-legit').checked) {
          document.querySelectorAll('.token.highlighted').forEach(token => token.classList.remove('highlighted'));
        }
        updateCheckButtonState();
      });
  
      document.getElementById('prev-email').addEventListener('click', () => {
        if (emailIndex > 0) {
          emailIndex--;
          fetchEmail(emailIndex);
          updateNavigationButtons();
        }
      });
  
      document.getElementById('next-email').addEventListener('click', () => {
        if (emailIndex < totalEmails - 1) {
          emailIndex++;
          fetchEmail(emailIndex);
          updateNavigationButtons();
        }
      });
  
      document.getElementById('check-email').addEventListener('click', handleCheckEmail);
  
      document.getElementById('continue-btn').addEventListener('click', () => {
        document.getElementById('info-box').style.display = 'none';
        document.getElementById('email-content').style.display = 'block';
        document.getElementById('nav-buttons').style.display = 'flex';
        document.getElementById('info-icon').style.display = 'inline-block';
      });
  
      document.getElementById('info-icon').addEventListener('click', toggleInfoBox);
  
      document.getElementById('language-switch').addEventListener('change', handleLanguageSwitch);
    };
  
    // Init setup
    const init = () => {
        updateNavigationButtons();
        fetchEmail(emailIndex);
        initializeEmailBoxClick();
        initEventListeners();
      };
      
    init();
  })();
  