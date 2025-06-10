class Translator {
    constructor() {
        this.translations = {};
        this.currentLang = 'en';
        this.loadTranslations();
    }

    async loadTranslations() {
        try {
            const response = await fetch(`/translations/${this.currentLang}.json`);
            this.translations = await response.json();
            this.translatePage();
        } catch (error) {
            console.error('Error loading translations:', error);
        }
    }

    async switchLanguage(lang) {
        this.currentLang = lang;
        await this.loadTranslations();
        document.documentElement.lang = lang;
        // document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr';
    }

    translate(key) {
        const keys = key.split('.');
        let value = this.translations;
        
        for (const k of keys) {
            if (value[k] === undefined) {
                console.warn(`Translation key not found: ${key}`);
                return key;
            }
            value = value[k];
        }
        
        return value;
    }

    translatePage() {
        // Translate all elements with data-i18n attribute
        document.querySelectorAll('[data-i18n]').forEach(element => {
            const key = element.getAttribute('data-i18n');
            const translation = this.translate(key);
            
            if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
                element.placeholder = translation;
            } else {
                // Replace \n with <br> tags for HTML elements
                element.innerHTML = translation.replace(/\n/g, '<br>');
            }
        });

        // Translate all elements with data-i18n-placeholder attribute
        document.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
            const key = element.getAttribute('data-i18n-placeholder');
            element.placeholder = this.translate(key);
        });

        // Translate all elements with data-i18n-key attribute (for dynamic article content)
        document.querySelectorAll('[data-i18n-key]').forEach(element => {
            const key = element.getAttribute('data-i18n-key');
            const translation = this.translate(key);
            if (translation) {
                element.innerHTML = translation.replace(/\n/g, '<br>');
            }
        });
    }
}

// Initialize translator
const translator = new Translator();

// Add event listener for the language select dropdown
document.addEventListener('DOMContentLoaded', () => {
    const languageSelect = document.getElementById('languageSelect');
    if (languageSelect) {
        languageSelect.value = translator.currentLang;
        languageSelect.addEventListener('change', (e) => {
            translator.switchLanguage(e.target.value);
        });
    }
}); 