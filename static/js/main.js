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
                setupFeedbackHandlers();
            },
        });
    });
    
    // Function to set up all feedback handlers
    function setupFeedbackHandlers() {
        // "Yes, it's correct" button
        $('.feedback-btn').click(function() {
            var feedback = $(this).data('feedback');
            var predictionId = $(this).data('prediction-id');
            
            // Hide feedback container
            $(this).closest('.feedback-container').hide();
            
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
                    
                    // Show feedback message
                    $('#feedback-message').html('<div class="alert alert-success">' + response.message + '</div>');
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
        
        // "No, it's incorrect" button - shows correction form
        $('[id^=show-correction-]').click(function() {
            var predictionId = $(this).attr('id').replace('show-correction-', '');
            
            // Hide feedback buttons, show correction form
            $(this).closest('.feedback-buttons').hide();
            $('#correction-form-' + predictionId).show();
        });
        
        // Submit correction
        $('.submit-correction').click(function() {
            var predictionId = $(this).data('prediction-id');
            var correctedClassIdx = $('#corrected-class-' + predictionId).val();
            
            // Hide correction form
            $(this).closest('.correction-form').hide();
            
            // Show loading
            $('.loader').show();
            
            // Send correction to server
            $.ajax({
                type: 'POST',
                url: '/correct',
                data: JSON.stringify({
                    'prediction_id': predictionId,
                    'corrected_class_idx': correctedClassIdx
                }),
                contentType: 'application/json',
                success: function(response) {
                    $('.loader').hide();
                    
                    // Show feedback message with the corrected class
                    $('#feedback-message').html(
                        '<div class="alert alert-success">' + 
                        response.message + 
                        '<br>Corrected to: <strong>' + response.corrected_class + '</strong>' +
                        '</div>'
                    );
                    $('#feedback-message').show();
                },
                error: function(xhr) {
                    $('.loader').hide();
                    let errorMsg = 'An error occurred while submitting your correction.';
                    
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