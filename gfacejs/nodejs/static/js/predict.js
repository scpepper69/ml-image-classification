const sleep = msec => new Promise(resolve => setTimeout(resolve, msec));

//------------------------------------------------
// Classification 
//------------------------------------------------
const CLASSES = {0:'RX-178', 1:'MSZ-006', 2:'RX-93', 3:'MS-06'}

//------------------------------------------------
// start button event
//------------------------------------------------
$("#start-button").click(function(){
    loadModel() ;
    startWebcam();
    //setInterval(predict, 1/10);
    setInterval(predict, 100);
});

//------------------------------------------------
// load model
//------------------------------------------------
let model;
async function loadModel() {
    console.log("AI model loading..");
    $("#console").html(`<x2>AI model loading...</x2>`);
    model=await tf.loadLayersModel(`http://localhost:8080/models/model.json`);
    console.log("AI Trained model loaded.");
    $("#console").html(`<x2>AI Trained model loaded.</x2>`);
};

//------------------------------------------------
// start web camera
//------------------------------------------------
var video;
function startWebcam() {
    console.log("video streaming start.");
    $("#console").html(`<x2>video streaming start.</x2>`);
    video = $('#main-stream-video').get(0);
    vendorUrl = window.URL || window.webkitURL;
    navigator.getMedia = navigator.getUserMedia ||
                         navigator.webkitGetUserMedia ||
                         navigator.mozGetUserMedia ||
                         navigator.msGetUserMedia;
    navigator.getMedia({
        video: true,
        audio: false
    }, function(stream) {
        localStream = stream;
        video.srcObject = stream;
        video.play();
    }, function(error) {
        alert("Something wrong with webcam!");
    });
}

//------------------------------------------------
// TensorFlow.js Predict
//------------------------------------------------
async function predict(){
    let tensor = captureWebcam();
    let prediction = await model.predict(tensor).data();
    let results = Array.from(prediction)
                .map(function(p,i){
    return {
        probability: Math.round(p*1000)/10,
        className: CLASSES[i]
    };
    }).sort(function(a,b){
        return b.probability-a.probability;
    }).slice(0,1);

    $("#console").empty();

    results.forEach(function(p){
        $("#imgText").html(`<x1>${p.className} (${p.probability.toFixed(1)} %) </x1>`);
        //$("#console").append(`<li>${p.className} : ${p.probability.toFixed(6)}</li>`);
        console.log(p.className,p.probability.toFixed(6))
    });

};

//------------------------------------------------
// capture streaming video 
// to a canvas object
//------------------------------------------------
function captureWebcam() {
    var canvas    = document.createElement("canvas");
    var context   = canvas.getContext('2d');
    canvas.width  = video.width;
    canvas.height = video.height;

    context.drawImage(video, 0, 0, video.width, video.height);
    tensor_image = preprocessImage(canvas);

    return tensor_image;
}

//------------------------------------------------
// TensorFlow.js Image to Tensor
//------------------------------------------------
function preprocessImage(image){
    let tensor = tf.browser.fromPixels(image).resizeNearestNeighbor([64,64]).toFloat();    
    let offset = tf.scalar(1);
    return tensor.div(offset).expandDims();
}

//------------------------------------------------
// clear button event
//------------------------------------------------
$("#clear-button").click(function clear() {
    location.reload();
});