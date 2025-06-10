document.addEventListener('DOMContentLoaded', function() {
    // Remove authentication check and keep only the core functionality
    
    // Get DOM elements
    const editButton = document.getElementById('editButton');
    const editModal = document.getElementById('editModal');
    const closeBtn = document.querySelector('.close-btn');
    const cancelBtn = document.querySelector('.cancel-btn');
    const saveBtn = document.getElementById('saveProfileBtn');
    const successMessage = document.getElementById('successMessage');

    // Form inputs
    const inputFullName = document.getElementById('inputFullName');
    const inputAge = document.getElementById('inputAge');
    const inputPhone = document.getElementById('inputPhone');
    const inputBloodType = document.getElementById('inputBloodType');
    const inputPregnancyTime = document.getElementById('inputPregnancyTime');
    const pregnancyTimeUnit = document.getElementById('pregnancyTimeUnit');
    
    // Display elements
    const displayFullName = document.getElementById('displayFullName');
    const displayAge = document.getElementById('displayAge');
    const displayPhone = document.getElementById('displayPhone');
    const displayBloodType = document.getElementById('displayBloodType');
    const displayPregnancyWeek = document.getElementById('displayPregnancyWeek');
    const pregnancyProgress = document.getElementById('pregnancyProgress');
    const currentWeek = document.getElementById('currentWeek');

    const resetButton = document.getElementById('resetButton');
    const form = document.getElementById('editProfileForm');

    // Initialize with default data
    const defaultData = {
        fullName: '-',
        age: '-',
        phone: '-',
        bloodType: '-',
        pregnancyTime: '0',
        pregnancyTimeUnit: 'weeks'
    };

    updateDisplayView(defaultData);

    // Health form inputs
    const inputBloodPressure = document.getElementById('inputBloodPressure');
    const inputBloodSugar = document.getElementById('inputBloodSugar');
    const inputWeight = document.getElementById('inputWeight');
    const inputSymptoms = document.getElementById('inputSymptoms');

    // Health display elements
    const displayBloodPressure = document.getElementById('displayBloodPressure');
    const displayBloodSugar = document.getElementById('displayBloodSugar');
    const displayWeight = document.getElementById('displayWeight');
    const displaySymptoms = document.getElementById('displaySymptoms');

    // Event Listeners
    editButton.addEventListener('click', function() {
        editModal.style.display = 'block';
        loadFormData();
    });

    closeBtn.addEventListener('click', function() {
        editModal.style.display = 'none';
    });

    cancelBtn.addEventListener('click', function() {
        editModal.style.display = 'none';
    });

    saveBtn.addEventListener('click', saveChanges);

    // Close modal when clicking outside
    window.addEventListener('click', function(e) {
        if (e.target == editModal) {
            editModal.style.display = 'none';
        }
    });

    function loadFormData() {
        const data = loadProfileData();
        if (data) {
            inputFullName.value = data.fullName || '';
            inputAge.value = data.age || '';
            inputPhone.value = data.phone || '';
            inputBloodType.value = data.bloodType || '';
            inputPregnancyTime.value = data.pregnancyTime || '';
            pregnancyTimeUnit.value = data.pregnancyTimeUnit || 'weeks';
        }
    }

    function validateForm() {
        let isValid = true;
        
        // Get error message elements
        const nameError = document.getElementById('nameError');
        const ageError = document.getElementById('ageError');
        const phoneError = document.getElementById('phoneError');
        const bloodError = document.getElementById('bloodError');
        const pregnancyError = document.getElementById('pregnancyError');

        // Reset all error messages
        [nameError, ageError, phoneError, bloodError, pregnancyError].forEach(error => {
            error.textContent = '';
            error.classList.remove('show');
        });
        
        // Validate Full Name
        if (!inputFullName.value.match(/^[A-Za-z\s]{2,}$/)) {
            nameError.textContent = 'Please enter a valid name (letters only)';
            nameError.classList.add('show');
            inputFullName.setCustomValidity('Invalid name');
            isValid = false;
        } else {
            inputFullName.setCustomValidity('');
        }

        // Validate Age
        const age = parseInt(inputAge.value);
        if (!age || age < 18 || age > 50) {
            ageError.textContent = 'Age must be between 18 and 50';
            ageError.classList.add('show');
            inputAge.setCustomValidity('Invalid age');
            isValid = false;
        } else {
            inputAge.setCustomValidity('');
        }

        // Validate Phone
        if (!inputPhone.value.match(/^\+?[\d\s-]{10,}$/)) {
            phoneError.textContent = 'Please enter a valid phone number';
            phoneError.classList.add('show');
            inputPhone.setCustomValidity('Invalid phone');
            isValid = false;
        } else {
            inputPhone.setCustomValidity('');
        }

        // Validate Blood Type
        if (!inputBloodType.value) {
            bloodError.textContent = 'Please select a blood type';
            bloodError.classList.add('show');
            inputBloodType.setCustomValidity('Invalid blood type');
            isValid = false;
        } else {
            inputBloodType.setCustomValidity('');
        }

        // Validate Pregnancy Time
        const pregnancyTime = parseInt(inputPregnancyTime.value);
        const timeUnit = pregnancyTimeUnit.value;

        if (!pregnancyTime || pregnancyTime < 0) {
            pregnancyError.textContent = 'Please enter a valid number';
            pregnancyError.classList.add('show');
            isValid = false;
        } else if (timeUnit === 'weeks') {
            if (pregnancyTime < 2 || pregnancyTime > 41) {
                pregnancyError.textContent = 'Pregnancy weeks must be between 2 and 41';
                pregnancyError.classList.add('show');
                isValid = false;
            }
        } else if (timeUnit === 'days') {
            if (pregnancyTime < 14 || pregnancyTime > 287) { // 2 weeks = 14 days, 41 weeks = 287 days
                pregnancyError.textContent = 'Pregnancy days must be between 14 and 287';
                pregnancyError.classList.add('show');
                isValid = false;
            }
        }

        return isValid;
    }

    // Calculate weeks from days
    function calculateWeeksFromDays(days) {
        return {
            weeks: Math.floor(days / 7),
            remainingDays: days % 7
        };
    }

    // Update validation for pregnancy time
    function validatePregnancyTime() {
        const value = parseInt(inputPregnancyTime.value);
        const unit = pregnancyTimeUnit.value;
        
        if (!value || value < 0) {
            pregnancyError.textContent = 'Please enter a valid number';
            pregnancyError.classList.add('show');
            return false;
        }

        if (unit === 'weeks' && value > 40) {
            pregnancyError.textContent = 'Weeks must be between 1 and 40';
            pregnancyError.classList.add('show');
            return false;
        }

        if (unit === 'days' && value > 280) { // 40 weeks * 7 days
            pregnancyError.textContent = 'Days must be between 1 and 280';
            pregnancyError.classList.add('show');
            return false;
        }

        pregnancyError.textContent = '';
        pregnancyError.classList.remove('show');
        return true;
    }

    // Update save changes function with validation
    function saveChanges() {
        if (!validateForm()) {
            return; // Don't save if validation fails
        }

        const profileData = {
            fullName: inputFullName.value,
            age: inputAge.value,
            phone: inputPhone.value,
            bloodType: inputBloodType.value,
            pregnancyTime: inputPregnancyTime.value,
            pregnancyTimeUnit: pregnancyTimeUnit.value
        };

        localStorage.setItem('profileData', JSON.stringify(profileData));
        updateDisplayView(profileData);
        editModal.style.display = 'none';
        showSuccessMessage();
    }

    function updateDisplayView(data) {
        displayFullName.textContent = data.fullName || '-';
        displayAge.textContent = data.age || '-';
        displayPhone.textContent = data.phone || '-';
        displayBloodType.textContent = data.bloodType || '-';
        
        // Calculate weeks and days
        let weeks = 0;
        let days = 0;

        if (data.pregnancyTimeUnit === 'days') {
            const calculated = calculateWeeksFromDays(parseInt(data.pregnancyTime) || 0);
            weeks = calculated.weeks;
            days = calculated.remainingDays;
        } else {
            weeks = parseInt(data.pregnancyTime) || 0;
        }

        // Update pregnancy week display
        displayPregnancyWeek.textContent = days > 0 ? 
            `${weeks} weeks, ${days} days` : 
            `${weeks} weeks`;
        
        // Update pregnancy tracker
        const totalWeeks = weeks + (days / 7);
        const progress = (totalWeeks / 40) * 100;
        pregnancyProgress.style.width = `${Math.min(progress, 100)}%`;
        currentWeek.textContent = Math.floor(totalWeeks);
    }

    function showSuccessMessage() {
        successMessage.style.display = 'flex';
        setTimeout(() => {
            successMessage.style.display = 'none';
        }, 3000);
    }

    // Initialize modal
    const confirmProfileResetModal = new bootstrap.Modal(document.getElementById('confirmProfileResetModal'));
    const confirmResetProfileBtn = document.getElementById('confirmProfileResetBtn');

    // Show modal on reset button click
    resetButton.addEventListener('click', function() {
        confirmProfileResetModal.show();
    });

    // Handle confirm reset
    confirmResetProfileBtn.addEventListener('click', function() {
        // Clear localStorage
        localStorage.removeItem('profileData');

        // Reset display values
        const defaultData = {
            fullName: '-',
            age: '-',
            phone: '-',
            bloodType: '-',
            pregnancyTime: '0',
            pregnancyTimeUnit: 'weeks'
        };
        updateDisplayView(defaultData);

        // Close modal
        confirmProfileResetModal.hide();

        // Show success message
        showSuccessMessage();
    });

    // Update loadProfileData function
    function loadProfileData() {
        // Load profile data
        const savedData = localStorage.getItem('profileData');
        if (savedData) {
            const profileData = JSON.parse(savedData);
            updateDisplayView(profileData);
            return profileData;
        }
        return null;
    }

    // Add real-time validation
    inputFullName.addEventListener('input', validateForm);
    inputAge.addEventListener('input', validateForm);
    inputPhone.addEventListener('input', validateForm);
    inputBloodType.addEventListener('change', validateForm);
    inputPregnancyTime.addEventListener('input', validateForm);
    pregnancyTimeUnit.addEventListener('change', validateForm);

    // Initialize profile data on page load
    loadProfileData();

    const saveHealthBtn = document.getElementById('saveHealthBtn');
    
    function validateHealthForm() {
        let isValid = true;
        
        // Reset all error states
        const errorMessages = document.querySelectorAll('.error-message');
        errorMessages.forEach(msg => {
            msg.textContent = '';
            msg.classList.remove('show');
        });
        
        const inputs = document.querySelectorAll('.form-control');
        inputs.forEach(input => {
            input.classList.remove('is-invalid');
        });
        
        // Validate Blood Pressure
        const systolic = parseInt(document.getElementById('inputSystolic').value);
        const diastolic = parseInt(document.getElementById('inputDiastolic').value);
        
        if (!systolic || systolic < 60 || systolic > 200 || !diastolic || diastolic < 40 || diastolic > 130) {
            document.getElementById('bpError').textContent = 'Invalid blood pressure range';
            document.getElementById('bpError').classList.add('show');
            document.getElementById('inputSystolic').classList.add('is-invalid');
            document.getElementById('inputDiastolic').classList.add('is-invalid');
            isValid = false;
        } else if (systolic <= diastolic) {
            document.getElementById('bpError').textContent = 'Systolic must be greater than diastolic';
            document.getElementById('bpError').classList.add('show');
            document.getElementById('inputSystolic').classList.add('is-invalid');
            document.getElementById('inputDiastolic').classList.add('is-invalid');
            isValid = false;
        }

        // Validate Blood Sugar
        const sugar = parseInt(inputBloodSugar.value);
        if (!sugar || sugar < 70 || sugar > 300) {
            document.getElementById('sugarError').textContent = 'Blood sugar must be between 70-180 mg/dL';
            document.getElementById('sugarError').classList.add('show');
            inputBloodSugar.classList.add('is-invalid');
            isValid = false;
        }

        // Validate Weight
        const weight = parseInt(inputWeight.value);
        if (!weight || weight < 40 || weight > 200) {
            document.getElementById('weightError').textContent = 'Weight must be between 40-200 kg';
            document.getElementById('weightError').classList.add('show');
            inputWeight.classList.add('is-invalid');
            isValid = false;
        }

        return isValid;
    }

    function saveHealthInfo() {
        if (!validateHealthForm()) {
            return;
        }

        const healthData = {
            bloodPressure: `${document.getElementById('inputSystolic').value}/${document.getElementById('inputDiastolic').value}`,
            bloodSugar: inputBloodSugar.value,
            weight: inputWeight.value,
            symptoms: inputSymptoms.value
        };

        localStorage.setItem('healthData', JSON.stringify(healthData));
        updateHealthDisplay(healthData);
        
        // Close modal using Bootstrap
        const modal = bootstrap.Modal.getInstance(document.getElementById('healthModal'));
        modal.hide();
        
        showSuccessMessage();
    }

    function updateHealthDisplay(data) {
        displayBloodPressure.textContent = data.bloodPressure || '-';
        displayBloodSugar.textContent = data.bloodSugar ? `${data.bloodSugar} Mg/DL` : '-';
        displayWeight.textContent = data.weight ? `${data.weight} Kg` : '-';
        displaySymptoms.textContent = data.symptoms || '-';
    }

    function loadHealthData() {
        const savedData = localStorage.getItem('healthData');
        if (savedData) {
            const healthData = JSON.parse(savedData);
            
            // Split blood pressure into systolic and diastolic
            if (healthData.bloodPressure) {
                const [systolic, diastolic] = healthData.bloodPressure.split('/');
                document.getElementById('inputSystolic').value = systolic;
                document.getElementById('inputDiastolic').value = diastolic;
            }

            updateHealthDisplay(healthData);
        }
    }

    // Add event listeners
    saveHealthBtn.addEventListener('click', saveHealthInfo);
    
    // Load health data on page load
    loadHealthData();

    const resetHealthBtn = document.getElementById('resetHealthBtn');
    const confirmResetBtn = document.getElementById('confirmResetBtn');
    const confirmResetModal = new bootstrap.Modal(document.getElementById('confirmResetModal'));

    resetHealthBtn.addEventListener('click', function() {
        // Show confirmation modal instead of browser confirm
        confirmResetModal.show();
    });

    confirmResetBtn.addEventListener('click', function() {
        // Reset form inputs
        document.getElementById('inputSystolic').value = '';
        document.getElementById('inputDiastolic').value = '';
        inputBloodSugar.value = '';
        inputWeight.value = '';
        inputSymptoms.value = '';

        // Clear health data from localStorage
        localStorage.removeItem('healthData');

        // Reset display values
        const defaultData = {
            bloodPressure: '-',
            bloodSugar: '-',
            weight: '-',
            symptoms: '-'
        };
        updateHealthDisplay(defaultData);

        // Close both modals
        confirmResetModal.hide();
        const healthModal = bootstrap.Modal.getInstance(document.getElementById('healthModal'));
        healthModal.hide();

        // Show success message
        showSuccessMessage();
    });

    const editHealthBtn = document.querySelector('.edit-health-btn');
    
    editHealthBtn.addEventListener('click', function() {
        // Get the current displayed values
        const currentBP = document.getElementById('displayBloodPressure').textContent;
        const currentSugar = document.getElementById('displayBloodSugar').textContent;
        const currentWeight = document.getElementById('displayWeight').textContent;
        const currentSymptoms = document.getElementById('displaySymptoms').textContent;

        // Populate blood pressure inputs
        if (currentBP !== '-') {
            const [systolic, diastolic] = currentBP.split('/');
            document.getElementById('inputSystolic').value = systolic.trim();
            document.getElementById('inputDiastolic').value = diastolic.trim();
        }

        // Populate blood sugar input (remove 'mg/DL' if present)
        if (currentSugar !== '-') {
            const sugarValue = currentSugar.replace(' Mg/DL', '').trim();
            document.getElementById('inputBloodSugar').value = sugarValue;
        }

        // Populate weight input (remove 'Kg' if present)
        if (currentWeight !== '-') {
            const weightValue = currentWeight.replace(' Kg', '').trim();
            document.getElementById('inputWeight').value = weightValue;
        }

        // Populate symptoms
        if (currentSymptoms !== '-') {
            document.getElementById('inputSymptoms').value = currentSymptoms;
        }
    });
});

// Tab switching functionality
const tabLinks = document.querySelectorAll('.nav-link');
const tabPanes = document.querySelectorAll('.tab-pane');

tabLinks.forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        
        // Remove active class from all links and panes
        tabLinks.forEach(l => l.classList.remove('active'));
        tabPanes.forEach(p => p.classList.remove('active'));
        
        // Add active class to clicked link and corresponding pane
        link.classList.add('active');
        const targetId = link.getAttribute('data-target');
        document.getElementById(targetId).classList.add('active');
    });
});

// Health form submission
const healthForm = document.getElementById('healthForm');
if (healthForm) {
    healthForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const healthData = {
            familyHistory: document.getElementById('familyHistory').value,
            currentConditions: document.getElementById('currentConditions').value
        };
        analyzeHealthRisks(healthData).then(predictions => {
            displayHealthPredictions(predictions);
        });
    });
}

function displayHealthPredictions(predictions) {
    const predictionsHtml = predictions.map(pred => 
        `<div class="prediction-card">
            <h5>${pred.condition}</h5>
            <p>Risk Level: ${pred.riskLevel}</p>
            <p>${pred.recommendation}</p>
        </div>`
    ).join('');
    
    document.getElementById('healthPredictions').innerHTML = predictionsHtml;
}

// Health Risk Predictor
function analyzeHealthRisks(healthData) {
    // This function would connect to your AI model
    // For now, we'll return a mock prediction
    return Promise.resolve([
        { condition: 'High Blood Pressure', riskLevel: 'High', recommendation: 'Monitor blood pressure regularly' },
        { condition: 'Diabetes', riskLevel: 'Medium', recommendation: 'Monitor blood sugar levels' },
        { condition: 'Obesity', riskLevel: 'High', recommendation: 'Consult a healthcare provider for weight management' }
    ]);
}
