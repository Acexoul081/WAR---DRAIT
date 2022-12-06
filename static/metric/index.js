
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
    console.log($(this).parent().serialize())
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
        }
    });
})

