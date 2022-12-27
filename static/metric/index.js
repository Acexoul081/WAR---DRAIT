function update_graph(metric){
    $.getJSON({
        url:'/metric', data:metric, success:function(result){
            
            Plotly.newplot('value', result, {})
            Plotly.newplot('loss', result, {})
        }
    })
}

$('button').on('click', function(){
    $('button').removeClass('selected');
    $(this).addClass('selected');
})

$('#static-btn').addClass('selected');

$('#static-btn').click(function(){
    // get metric data here
    $.ajax({
        type: "POST",
        url: '/static',
        success: function(result) {
            console.log('update success')
        }
    });
});

$('#dynamic-btn').click(function(e){
    $.ajax({
        type: "POST",
        url: '/dynamic',
        success: function(result) {
            console.log('update success')
        }
    });
});

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
        $.ajax({
            type: "POST",
            url: '/update-cron',
            data: $(this).parent().serialize(),
            // data: { 
            //     cron: $(this).val(),
            //     // access_token: $("#access_token").val()
            // },
            success: function(result) {
                console.log('update success')
                window.location.reload()
            }
        });
    }else{
        alert("Invalid Cron Format")
    }
})

$('#cron_input').on('input', function(e){
    // console.log($(this).val());
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