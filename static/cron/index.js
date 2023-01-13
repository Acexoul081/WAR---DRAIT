const toastTrigger = document.getElementById('liveToastBtn')
const toastLiveExample = document.getElementById('liveToast')
sessionStorage.reloadAfterPageLoad = false

$('.update-cron-btn').click(function(e){
    e.preventDefault();
    var cron = $(this).parent().serializeArray()
    isCronValid = false
    $.each(cron, function(i, field) {
        if (field.name === 'new-cron') {
            isCronValid = checkCron(field.value)
        }
    });
    if (isCronValid) {
        $('.update-cron-btn').attr("disabled", true)
        $.ajax({
            type: "POST",
            url: `/cron/${$(this).val()}`,
            data: $(this).parent().serialize(),
            // data: { 
            //     cron: $(this).val(),
            //     // access_token: $("#access_token").val()
            // },
            success: function(result) {
                sessionStorage.reloadAfterPageLoad = true;
                // window.location.reload()
                console.log(result)
            }
        });
    }else{
        alert("Invalid Cron Format")
    }
})

$( function () {
    sessionStorage.reloadAfterPageLoad = false
    console.log(sessionStorage.reloadAfterPageLoad)
    if ( sessionStorage.reloadAfterPageLoad ) {
        $('.update-cron-btn').attr("disabled", false)
        const toast = new bootstrap.Toast(toastLiveExample)
        toast.show()
        sessionStorage.reloadAfterPageLoad = false;
    }
})

$('#cron_input').on('input', function(e){
    // console.log($(this).val());
    console.log(checkIfStringHasSpecialChar($(this).val()));
    try {
        result = cronstrue.toString($(this).val())
    } catch (error) {
        // console.log('error');
        result = undefined
        $(this).removeClass('border-0');
        $(this).addClass('border border-danger border-1');
    }
    // console.log(result);
    if (result !== undefined) {
        $('#cron_label').text("\""+result+"\"");
        $(this).removeClass('border border-danger border-1');
        $(this).addClass('border-0');
    }
})

$(document).ready(function() {
    result = cronstrue.toString("* * * * *")
    $('#cron_label').text("\""+result+"\"");
});

function checkIfStringHasSpecialChar(_string)
{
    let spChars = /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]+/;
    if(spChars.test(_string)){
      return true;
    } else {
      return false;
    }
}

function checkCron(cron){
    try {
        result = cronstrue.toString(cron)
    } catch (error) {
        result = undefined
        return false
    }
    if (result !== undefined) {
        return true
    }
}
