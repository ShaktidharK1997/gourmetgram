$(document).ready(function () {
    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        $('#feedback-message').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').html(data);
                $('#result').show();
                
                // Set up event handlers for feedback buttons
                setupFeedbackButtons();
            },
        });
    });

   // Function to set up feedback button handlers
    function setupFeedbackButtons() {
    $('.feedback-btn').click(function() {
        var feedback = $(this).data('feedback');
        var predictionId = $(this).data('prediction-id');
        var $this = $(this);
        
        // Disable both feedback buttons
        $('.feedback-btn').prop('disabled', true);
        
        // Visual feedback for selected button
        if (feedback === 'yes') {
            $this.find('i').removeClass('bi-hand-thumbs-up').addClass('bi-hand-thumbs-up-fill');
            $this.addClass('text-success');
        } else {
            $this.find('i').removeClass('bi-hand-thumbs-down').addClass('bi-hand-thumbs-down-fill');
            $this.addClass('text-danger');
        }
        
        // Add a subtle animation
        $this.addClass('animate__animated animate__pulse');
        
        // Send feedback to server
        $.ajax({
            type: 'POST',
            url: '/feedback',
            data: JSON.stringify({
                'prediction_id': predictionId,
                'feedback': feedback
            }),
            contentType: 'application/json',
            success: function(response) {
                // Replace the question with a thank you message
                $this.closest('.d-flex').html('<small class="text-success">Thank you for your feedback!</small>');
                
                // If there's additional information to display
                if (response.task_created) {
                    $('#feedback-message').html('<div class="alert alert-success small mt-2">Your feedback has been sent to our review team.</div>');
                    $('#feedback-message').show();
                }
            },
            error: function(xhr) {
                // Re-enable buttons on error
                $('.feedback-btn').prop('disabled', false);
                $this.removeClass('text-success text-danger');
                $this.find('i').removeClass('bi-hand-thumbs-up-fill bi-hand-thumbs-down-fill')
                     .addClass(feedback === 'yes' ? 'bi-hand-thumbs-up' : 'bi-hand-thumbs-down');
                
                // Show error message
                let errorMsg = 'An error occurred while submitting your feedback.';
                if (xhr.responseJSON && xhr.responseJSON.message) {
                    errorMsg += ' ' + xhr.responseJSON.message;
                }
                
                $('#feedback-message').html('<div class="alert alert-danger small">' + errorMsg + '</div>');
                $('#feedback-message').show();
            }
        });
    });
    }
});