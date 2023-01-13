const toastTrigger = document.getElementById('liveToastBtn')
const toastLiveExample = document.getElementById('liveToast')
sessionStorage.reloadAfterPageLoad = false

function refreshCronButton(result){
    console.log('update success')
    console.log(result.crons)
    var cronContainer = document.getElementById("cron-section");
    cronContainer.innerHTML = ''
    
    result.crons.forEach(element => {
        console.log(result.metric)
        let section = document.createElement('cron-section');
        section.innerHTML = `
        <div class="card mb-3">
            <div class="card-header">
                ${element.schedule}
                <div class="vr mx-1"></div>
                ${element.schedule_readable}
            </div>
            <div class="card-body">
                <h5 class="card-title">${element.schedule_readable}</h5>
                <p class="card-text">
                    ${element.job_description}
                </p>
                <form class="row g-2">
                    <input type="text" class="form-control text-center" name="new-cron" placeholder="${element.schedule}" value="${element.schedule}">
                    <input type="hidden" name="prev-cron" value='${element.job_detail}'>
                    <button type="button" value='${result.metric}' class="btn btn-primary update-cron-btn col-sm-2 mx-auto">Update Schedule</button>
                </form>
            </div>
        </div>
        `;
        cronContainer.appendChild(section);
    });
    $('.update-cron-btn').click(function(e){
        e.preventDefault();
        console.log("masuk");
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
                success: function(result) {
                    refreshCronButton(result)
                    alert("Update Schedule Success!")
                }
            });
        }else{
            alert("Invalid Cron Format")
        }
    })
}

$('.update-cron-btn').click(function(e){
    e.preventDefault();
    console.log("masuk");
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
            success: function(result) {
                refreshCronButton(result)
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
