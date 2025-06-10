const sign_in_btn = document.querySelector("#sign-in-btn");
const sign_up_btn = document.querySelector("#sign-up-btn");
const container = document.querySelector(".container");

sign_up_btn.addEventListener("click", () => {
  container.classList.add("sign-up-mode");
});

sign_in_btn.addEventListener("click", () => {
  container.classList.remove("sign-up-mode");
});

// --- Backend Integration for Login & Signup ---

document.addEventListener('DOMContentLoaded', function() {
  // Select forms
  const loginForm = document.querySelector('.sign-in-form');
  const signupForm = document.querySelector('.sign-up-form');

  // LOGIN HANDLER
  loginForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    const usernameOrEmail = document.getElementById('login-identifier').value.trim();
    const password = document.getElementById('login-password').value;

    if (!usernameOrEmail || !password) {
      alert('Please enter both username/email and password.');
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ identifier: usernameOrEmail, password })
      });
      const data = await response.json();
      if (response.ok) {
        localStorage.setItem('token', data.token);
        localStorage.setItem('userData', JSON.stringify(data.user));
        window.location.href = '/profile';
      } else {
        alert(data.error || 'Login failed.');
      }
    } catch (err) {
      alert('Network error. Please try again.');
    }
  });

  // SIGNUP HANDLER
  signupForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    const name = document.getElementById('signup-username').value.trim();
    const email = document.getElementById('signup-email').value.trim();
    const password = document.getElementById('signup-password').value;
    // Generate a unique phone number for each signup
    const phone = '01' + Math.floor(Math.random() * 1000000000).toString().padStart(9, '0');

    if (!name || !email || !password) {
      alert('Please fill all fields.');
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/api/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, password, phone })
      });
      const data = await response.json();
      if (response.ok) {
        alert('Signup successful! Please log in.');
        document.querySelector('#sign-in-btn').click();
      } else {
        alert(data.error || 'Signup failed.');
      }
    } catch (err) {
      alert('Network error. Please try again.');
    }
  });
});
