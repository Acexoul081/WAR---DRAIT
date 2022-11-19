
function update_graph(metric){
    $.getJSON({
        url:'/metric', data:metric, success:function(result){
            
            Plotly.newplot('value', result, {})
            Plotly.newplot('loss', result, {})
        }
    })
}

$('#static-btn').click(function(){
    // console.log(e)
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

