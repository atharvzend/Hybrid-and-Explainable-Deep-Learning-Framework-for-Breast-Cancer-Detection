document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    const loginBtn = document.getElementById('loginBtn');
    const btnText = loginBtn.querySelector('.btn-text');
    const btnLoader = loginBtn.querySelector('.btn-loader');
    const errorMessage = document.getElementById('errorMessage');
    const usernameInput = document.getElementById('username');
    const passwordInput = document.getElementById('password');

    // Handle form submission
    loginForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Get form values
        const username = usernameInput.value.trim();
        const password = passwordInput.value.trim();

        // Validate inputs
        if (!username || !password) {
            showError('Please enter both username and password');
            return;
        }

        // Show loading state
        setLoading(true);
        hideError();

        try {
            // Make API request
            const response = await fetch('/authenticate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: username,
                    password: password
                })
            });

            const data = await response.json();

            if (response.ok && data.success) {
                // Login successful
                showSuccess('Login successful! Redirecting...');
                
                // Redirect after short delay
                setTimeout(() => {
                    window.location.href = '/home';
                }, 1000);
            } else {
                // Login failed
                showError(data.message || 'Invalid credentials. Please try again.');
                setLoading(false);
            }
        } catch (error) {
            console.error('Login error:', error);
            showError('An error occurred. Please try again later.');
            setLoading(false);
        }
    });

    // Set loading state
    function setLoading(isLoading) {
        loginBtn.disabled = isLoading;
        
        if (isLoading) {
            btnText.style.display = 'none';
            btnLoader.style.display = 'block';
        } else {
            btnText.style.display = 'block';
            btnLoader.style.display = 'none';
        }
    }

    // Show error message
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        errorMessage.style.animation = 'shake 0.5s';
        
        // Remove animation class after it completes
        setTimeout(() => {
            errorMessage.style.animation = '';
        }, 500);
    }

    // Hide error message
    function hideError() {
        errorMessage.style.display = 'none';
    }

    // Show success message (reuse error message styling)
    function showSuccess(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        errorMessage.style.background = '#D4EDDA';
        errorMessage.style.borderColor = '#C3E6CB';
        errorMessage.style.color = '#155724';
    }

    // Clear error on input
    usernameInput.addEventListener('input', hideError);
    passwordInput.addEventListener('input', hideError);

    // Demo credential quick fill (for development)
    document.querySelectorAll('.demo-item code').forEach(item => {
        item.style.cursor = 'pointer';
        item.title = 'Click to auto-fill';
        
        item.addEventListener('click', function() {
            const text = this.textContent.trim();
            const [user, pass] = text.split(' / ');
            
            if (user && pass) {
                usernameInput.value = user;
                passwordInput.value = pass;
                usernameInput.focus();
                
                // Show visual feedback
                this.style.background = '#E6F4F3';
                setTimeout(() => {
                    this.style.background = '';
                }, 300);
            }
        });
    });

    // Enter key handling
    passwordInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            loginForm.dispatchEvent(new Event('submit'));
        }
    });

    // Focus first input on page load
    usernameInput.focus();
});

// Add shake animation for errors
const style = document.createElement('style');
style.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
        20%, 40%, 60%, 80% { transform: translateX(5px); }
    }
`;
document.head.appendChild(style);