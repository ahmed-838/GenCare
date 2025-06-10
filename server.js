const express = require('express');
const path = require('path');
const articles = require('./data/articles.js');
const babyNames = require('./data/babynames.js');

// Create Express app
const app = express();

// Set up static file serving
app.use('/css', express.static(path.join(__dirname, 'node_modules/bootstrap/dist/css')));
app.use('/animate', express.static(path.join(__dirname, 'node_modules/animate.css')));
app.use('/bootstrap', express.static(path.join(__dirname, 'node_modules/bootstrap/dist')));
app.use('/popper', express.static(path.join(__dirname, 'node_modules/@popperjs/core/dist/umd')));

app.use(express.static(path.join(__dirname, 'public')));

// Set up EJS as the view engine
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));


// Define route for the homepage
app.get('/', (req, res) => {
    res.render('home', { babyNames: babyNames });
});

// Dynamic route for articles based on week
app.get('/week/:week', (req, res) => {
    const { week } = req.params;
    const article = articles.find(a => a.week === parseInt(week));
  
    if (article) {
      res.render('pregnancy-article', { article });
    } else {
      res.status(404).send('Article not found');
    }
  });

//main route for baby names
app.get('/baby-names', (req, res) => {
  const defaultLetter = babyNames[0]; 

  res.render('baby-names', { 
    babyNames: babyNames,
    currentLetter: defaultLetter.letter,
    currentNames: {
      boys: defaultLetter.boys,
      girls: defaultLetter.girls
    }
  });
});

// Route for specific letter
app.get('/baby-names/:letter', (req, res) => {
  const requestedLetter = req.params.letter.toUpperCase();
  const letterData = babyNames.find(item => item.letter === requestedLetter);
  
  if (letterData) {
    res.render('baby-names', { 
      babyNames: babyNames,
      currentLetter: letterData.letter,
      currentNames: {
        boys: letterData.boys,
        girls: letterData.girls
      }
    });
  } else {
    // If letter not found, redirect to main page
    res.redirect('/baby-names');
  }
});

// Route for the AI model landing page
app.get('/ai-model', (req, res) => {
    res.sendFile(path.join(__dirname, 'ai-model', 'landing.html'));
});

// Serve static files from the ai-model directory
app.use('/ai-model', express.static(path.join(__dirname, 'ai-model')));


// Route for the AI model main page
app.get('/ai-model/index.html', (req, res) => {
    res.sendFile(path.join(__dirname, 'ai-model', 'index.html'));
});

app.use('/disease-pages', express.static(path.join(__dirname, 'disease')));


// Serve static files from the login directory
app.use('/login', express.static(path.join(__dirname, 'login')));


app.get('/profile', (req, res) => {
  res.sendFile(path.join(__dirname, 'profile page', 'landing.html'));
});

app.use('/profile', express.static(path.join(__dirname, 'profile page')));

app.get('/profile/index.html', (req, res) => {
    res.sendFile(path.join(__dirname, 'profile page', 'index.html'));
});

module.exports = app;
