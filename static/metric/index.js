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
    $.ajax({
        type: "POST",
        url: '/static',
        data: {'metric':$(this).val()},
        success: function(result) {
            value_update = {x: [JSON.parse(result.x_value), JSON.parse(result.x_value_anom)], y: [JSON.parse(result.y_value), JSON.parse(result.y_value_anom)]}
            loss_update = {x: [JSON.parse(result.x_loss), JSON.parse(result.x_loss_anom)], y: [JSON.parse(result.y_loss), JSON.parse(result.y_loss_anom)]}
            Plotly.update('value', value_update, {})
            Plotly.update('loss', loss_update, {})
        }
    });
});

$('#dynamic-btn').click(function(e){
    console.log($(this).val())
    $.ajax({
        type: "POST",
        url: '/dynamic',
        data: {'metric':$(this).val()},
        success: function(result) {
            value_update = {x: [JSON.parse(result.x_value), JSON.parse(result.x_value_anom)], y: [JSON.parse(result.y_value), JSON.parse(result.y_value_anom)]}
            loss_update = {x: [JSON.parse(result.x_loss), JSON.parse(result.x_loss_anom)], y: [JSON.parse(result.y_loss), JSON.parse(result.y_loss_anom)]}
            Plotly.update('value', value_update, {})
            Plotly.update('loss', loss_update, {})
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