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
            
            // Hide feedback buttons after selection
            $('.feedback-buttons').hide();
            
            // Show loading
            $('.loader').show();
            
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
                    $('.loader').hide();
                    
                    let message = response.message;
                    
                    if (response.task_created) {
                        message += ' Your feedback has been sent to our review team.';
                    }
                    
                    // Show feedback message
                    $('#feedback-message').html('<div class="alert alert-success">' + message + '</div>');
                    $('#feedback-message').show();
                },
                error: function(xhr) {
                    $('.loader').hide();
                    let errorMsg = 'An error occurred while submitting your feedback.';
                    
                    if (xhr.responseJSON && xhr.responseJSON.message) {
                        errorMsg += ' ' + xhr.responseJSON.message;
                    }
                    
                    $('#feedback-message').html('<div class="alert alert-danger">' + errorMsg + '</div>');
                    $('#feedback-message').show();
                }
            });
        });
    }
});