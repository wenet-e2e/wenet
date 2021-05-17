
var recoder;
var sample_rate = 16000;
var bit_rate = 16;

function RecordStart() {
    $('#stop').attr("disabled", false);
    $('#record').attr("disabled", true);
    $("#recording").html("");
    if (recoder) {
        recoder.close();
    } else {
        recoder = Recorder({
            type:"wav",sampleRate:sample_rate,bitRate:bit_rate
            ,onProcess:function(buffers,powerLevel,bufferDuration,bufferSampleRate){
            }
        });
        recoder.open(function(){//打开麦克风授权获得相关资源
            recoder.start();//开始录音
        },function(msg,isUserNotAllow){
            alert((isUserNotAllow?"UserNotAllow，":"")+"无法录音:"+msg, 1);
        });
    }
}

function RecordStop() {
    recoder.stop(function(blob,duration){
		recoder.close(); //释放录音资源，当然可以不释放，后面可以连续调用start；但不释放时系统或浏览器会一直提示在录音，最佳操作是录完就close掉
        var recording = document.getElementById("recording");
        var url = URL.createObjectURL(blob);
        var au = document.createElement('audio');
        var hf = document.createElement('a');
        au.controls = true;
        au.src = url;
        hf.href = url;
        hf.download = new Date().toISOString() + '.wav';
        hf.innerHTML = hf.download;
        recording.appendChild(au);
        recording.appendChild(hf);
        recognize(blob, "/record");
		recoder=null;
	},function(msg){
		alert("录音失败:"+msg, 1);
		recoder.close();//可以通过stop方法的第3个参数来自动调用close
		recoder=null;
	});
    $('#stop').attr("disabled", true);
    $('#record').attr("disabled", false);
}


function recognize(blob, url) {
    if ((blob.size < 100)) {
        alert("Need record audio");
        recoder.close();
        return;
    }
    var formData = new FormData();
    formData.append('cuid', 'test');
    formData.append('audio', blob);
    console.log(formData);
    $.ajax({
        url: url,
        type: "HTTP",
        method: "POST",
        data: formData,
        contentType: false,
        cache: false,
        processData: false,
        success: function(req) {
            var ret = JSON.parse(req);
            console.log(ret);
            if (Number(ret['ret_code']) === 1){
                $('#result').html(ret['result']);
            }
            else {
                error_msg(req);
            }
        },
        error: function(){
            alert('Recognize failed');
        }
    })
    recoder.close();
}

$(function() {
    M.AutoInit();
    $('#stop').attr("disabled", true);
    $('#record').attr("disabled", false);
});