$(document).ready(function() {
	$('.sign-up-form').on('submit', submit);
	$('.reset-button').on('click', reset);
});
function submit(e) {
	$('.sign-up-container').addClass('submitted');
	e.preventDefault();
}
function reset() {
	$('.sign-up-container').removeClass('submitted');
	$('.sign-up-form')[0].reset();
}
$(document).ready(function() {
	$('.sign-up-form').on('submit', submit);
	$('.reset-button').on('click', reset);
  });
  function submit(e) {
	e.preventDefault(); 
	$('.sign-up-container').addClass('submitted');
	
	// show Loader
	setTimeout(function() {
      $('.loader').fadeIn();
	}, 2000);

	setTimeout(function() {
	  $('.loader').fadeOut(); 
	  $('.success-message').fadeIn(); 
	}, 8000); 
	
  }
  function reset() {
	$('.sign-up-container').removeClass('submitted');
	$('.sign-up-form')[0].reset();
  
	$('.loader').hide(); 
	$('.success-message').hide(); 
  }