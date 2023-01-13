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
