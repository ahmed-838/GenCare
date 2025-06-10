const mediaQuery = window.matchMedia("(min-width: 320px) and (max-width: 576px)");

// Function to modify attributes
function modifyAttributes(e) {
  const swiperContainer = document.querySelector("swiper-container");
  if (e.matches) {
    swiperContainer.setAttribute("slides-per-view", "3");
  } else {
    swiperContainer.setAttribute("slides-per-view", "6");
  }
}

// Add listener for changes in screen size
mediaQuery.addEventListener("change", modifyAttributes);
modifyAttributes(mediaQuery);

