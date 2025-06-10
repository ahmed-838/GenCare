handleClick = (ele)=>{
    var navlist = document.getElementsByClassName("nav-link")
    for(let i = 0 ; i < navlist.length ; i++){
        navlist[i].classList.remove("active")
    }
    ele.classList.add("active")
}
var nav = document.getElementById('nav')
window.addEventListener('scroll' , ()=>{
    if(document.documentElement.scrollTop>0){
        nav.classList.add("nav-scroll")

    }else{
        nav.classList.remove('nav-scroll')
    }
})
document.addEventListener("DOMContentLoaded", function() {
    // Function to add animation class to the active slide's h1
    function addAnimationToActiveSlide() {
        // Remove the animation class from all h1 elements first
        document.querySelectorAll('.carousel-item h1').forEach(h1 => {
            h1.classList.remove('animate__animated', 'animate__fadeInDown','opacity-100');
        });
        
        // Find the active carousel item and add the class to its h1
        const activeSlide = document.querySelector('.carousel-inner .carousel-item.active');
        if (activeSlide) {
            const h1 = activeSlide.querySelector('.card-body h1');
            if (h1) {
                h1.classList.add('animate__animated', 'animate__fadeInDown','opacity-100');
            }
        }
    }

    // Initialize the animation for the first active slide
    addAnimationToActiveSlide();

    // Listen for the Bootstrap carousel slide event and call the function
    document.getElementById('carouselExample').addEventListener('slid.bs.carousel', addAnimationToActiveSlide);
});

revealFunction = (ele) =>{
    const rect = ele.getBoundingClientRect();
    return(
        rect.bottom > 0 &&
        rect.top < (window.innerHeight - 50)
    )
}

var h2 = document.querySelectorAll('h2')
console.log(h2)
var cardArticle = document.querySelectorAll('#articles .card,#aiModel .card')
document.addEventListener('scroll' , ()=>{
    for(let i = 1 ; i < h2.length ; i++  ){
        if(revealFunction(h2[i])){
            h2[i].classList.add('animate__animated','animate__fadeInLeft','opacity-100')
            h2[i].classList.remove('opacity-0')
        }
    }
    for(let i = 0 ; i < cardArticle.length ; i++){
        if(revealFunction(cardArticle[i])){
            cardArticle[i].classList.add('animate__animated','animate__fadeInRight','opacity-100')
            cardArticle[i].classList.remove('opacity-0')
        }
    }
})

// new updates
// Define the media query range
const mediaQuery = window.matchMedia("(min-width: 320px) and (max-width: 576px)");

// Function to modify attributes
function modifyAttributes(e) {
  const imgElement = document.querySelector("#pregnancy-weeks swiper-container"); // Target your element

  if (e.matches) {
    // If the screen is within the range
    imgElement.setAttribute("slides-per-view", 3); // Example of modifying the src attribute
    // imgElement.setAttribute("alt", "New Alt Text");  // Example of modifying another attribute
  } else {
    // Revert to original attributes if needed
    imgElement.setAttribute("slides-per-view", 6);
    // imgElement.setAttribute("alt", "Original Alt Text");
  }
}

// Add listener for changes in screen size
mediaQuery.addEventListener("change", modifyAttributes);

// Initial check
modifyAttributes(mediaQuery);

function toggleNotifications(event) {
  event.preventDefault();
  const dropdown = document.getElementById('notificationDropdown');
  dropdown.classList.toggle('show');
}

// Hide notification dropdown when clicking outside
document.addEventListener('click', function(event) {
  const bell = document.getElementById('notificationBell');
  const dropdown = document.getElementById('notificationDropdown');
  if (!bell.contains(event.target) && !dropdown.contains(event.target)) {
    dropdown.classList.remove('show');
  }
});

function updateNotificationCount() {
  const notificationList = document.getElementById('notificationList');
  const notifCount = document.getElementById('notifCount');
  const notifications = notificationList.getElementsByTagName('li');
  // Don't count items with the text-muted class (empty state message)
  const count = Array.from(notifications).filter(item => !item.classList.contains('text-muted')).length;
  
  if (count > 0) {
    notifCount.textContent = count;
    notifCount.style.display = 'block';
  } else {
    notifCount.style.display = 'none';
  }
}

function clearNotifications() {
  const notifList = document.getElementById('notificationList');
  notifList.innerHTML = `<li class="px-3 py-2 text-center text-muted">No notifications</li>`;
  updateNotificationCount();
}

// Call this when the DOM is loaded to set initial count
document.addEventListener('DOMContentLoaded', function() {
  updateNotificationCount();
});
