var annotations = {};
var other = {};

function mark_success(word_id) {
    $("li#word_"+word_id).addClass("done");
}

function mark_error(word_id) {
    $("li#word_"+word_id).addClass("error");
}

function unmark(word_id) {
    $("li#word_"+word_id).removeClass("error");
    $("li#word_"+word_id).removeClass("done");
}

function update_annotation(word_id, analysis, is_other) {
    unmark(word_id);
    var data = {sentence_id: sentence_id, word_id: word_id, analysis: analysis};
    console.log(data);
    $.post(app_url, data, function(result) {
        if(result == "OK") {
            console.log("success");
            mark_success(word_id);
            if(is_other === true){
                annotations[word_id] = "__other__";
                other[word_id] = analysis
            }
            else
                annotations[word_id] = analysis;
        }
        else {
            console.error(analysis+" -> "+result);
            mark_error(word_id);
        }
    }).error(function(e) {
        console.error(analysis+" -> "+e);
        mark_error(word_id);
    })
}

$(function() {
    $("input[type=radio]").click(function() {
        console.log(this.value);
        var v = this.value.split("|");
        var word_id = parseInt(v[0]);
        var analysis = v[1];
        if(annotations[word_id] != analysis) {
            if(analysis == "__other__")
                update_annotation(word_id, $("input[name="+word_id+"]").val(), true);
            else
                update_annotation(word_id, analysis);
        }
    });

    $("input[type=text]").change(function() {
        console.log("other:"+this.value);
        var word_id = parseInt(this.name);
        if(annotations[word_id] == "__other__")
            update_annotation(word_id, this.value, true);
    }).keydown(function() {
        unmark(this.name);
    }).keyup(function() {
        if(other[parseInt(this.name)] == this.value)
            mark_success(this.name); // FIXME should revert to error if was error
    });
});
