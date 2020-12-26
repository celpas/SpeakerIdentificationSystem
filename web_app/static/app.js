// webkitURL is deprecated but nevertheless
URL = window.URL || window.webkitURL;

var gumStream; 						// stream from getUserMedia()
var rec; 							// Recorder.js object
var input; 							// MediaStreamAudioSourceNode we'll be recording

// shim for AudioContext when it's not avb. 
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext // audio context to help us record

var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");

// add events to those 2 buttons
recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);

// status
var uploading = false

// counter
var totalSeconds = 0
var counter = null

function myCounter() {
	++totalSeconds;
	document.getElementById("seconds").innerHTML = totalSeconds;
}

function startRecording() {
	console.log("recordButton clicked");

	/*
		Simple constraints object, for more advanced audio features see
		https://addpipe.com/blog/audio-constraints-getusermedia/
	*/
    
    var constraints = { audio: true, video:false }

 	/*
    	Disable the record button until we get a success or fail from getUserMedia() 
	*/

	recordButton.disabled = true;
	stopButton.disabled = false;

	/*
    	We're using the standard promise based getUserMedia() 
    	https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
	*/

	navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
		console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

		/*
			create an audio context after getUserMedia is called
			sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
			the sampleRate defaults to the one set in your OS for your playback device

		*/
		audioContext = new AudioContext();

		//update the format 
		document.getElementById("formats").innerHTML="Infos: 1 channel pcm @ "+audioContext.sampleRate/1000+"kHz - "

		//update counter
		document.getElementById("seconds").innerHTML = 0
		document.getElementById("counter").className = 'd-inline font-weight-bold'
		totalSeconds = 0
		counter = setInterval(myCounter, 1000);

		/*  assign to gumStream for later use  */
		gumStream = stream;
		
		/* use the stream */
		input = audioContext.createMediaStreamSource(stream);

		/* 
			Create the Recorder object and configure to record mono sound (1 channel)
			Recording 2 channels  will double the file size
		*/
		rec = new Recorder(input,{numChannels:1})

		//start the recording process
		rec.record()

		console.log("Recording started");

	}).catch(function(err) {
	  	//enable the record button if getUserMedia() fails
    	recordButton.disabled = false;
    	stopButton.disabled = true;
	});
}

function stopRecording() {
	console.log("stopButton clicked");

	// Disable counter
	clearInterval(counter);

	// Disable the stop button, enable the record too allow for new recordings
	stopButton.disabled = true;
	recordButton.disabled = false;
	
	// Tell the recorder to stop the recording
	rec.stop();

	// Stop microphone access
	gumStream.getAudioTracks()[0].stop();

	// Create the wav blob and pass it on to createDownloadLink
	rec.exportWAV(createDownloadLink);
}

function createDownloadLink(blob) {
	var url = URL.createObjectURL(blob);
	var audio_element = document.createElement('audio');
	var li = document.createElement('li');
	var download_button = document.createElement('a');

	// Filename equals to the unix timestamp
	var unix_timestamp = Date.now();

	// Current date
	var currentDate = new Date();
	var currentDateString = currentDate.toLocaleDateString('it-IT');
	var currentTimeString = currentDate.toLocaleTimeString('it-IT');

	// Current date container
	var date_container = document.createElement('p');
	date_container.className = 'font-weight-bold'
	date_container.append(document.createTextNode(currentDateString + ' ' + currentTimeString))

	// Set ID of the <li> element
	li.id = unix_timestamp
	li.className = 'list-group-item'

	// Add controls to the <audio> element
	audio_element.controls = true;
	audio_element.src = url;

	// Save to disk download_link
	download_button.href = url;
	download_button.download = unix_timestamp+".wav";
	download_button.innerHTML = "<svg width=\"1em\" height=\"1em\" viewBox=\"0 0 16 16\" class=\"bi bi-cloud-arrow-down-fill\" fill=\"currentColor\" xmlns=\"http://www.w3.org/2000/svg\">\n" +
		"  <path fill-rule=\"evenodd\" d=\"M8 2a5.53 5.53 0 0 0-3.594 1.342c-.766.66-1.321 1.52-1.464 2.383C1.266 6.095 0 7.555 0 9.318 0 11.366 1.708 13 3.781 13h8.906C14.502 13 16 11.57 16 9.773c0-1.636-1.242-2.969-2.834-3.194C12.923 3.999 10.69 2 8 2zm2.354 6.854l-2 2a.5.5 0 0 1-.708 0l-2-2a.5.5 0 1 1 .708-.708L7.5 9.293V5.5a.5.5 0 0 1 1 0v3.793l1.146-1.147a.5.5 0 0 1 .708.708z\"/>\n" +
		"</svg> Download";
	download_button.className = "btn btn-secondary btn-mm mr-2"

	// Response
	var response_container = document.createElement('div');
	response_container.className = 'response_container mt-3'

	// Upload
	var upload_button = document.createElement('button');
	upload_button.innerHTML = "<svg width=\"1em\" height=\"1em\" viewBox=\"0 0 16 16\" class=\"bi bi-cloud-arrow-up-fill\" fill=\"currentColor\" xmlns=\"http://www.w3.org/2000/svg\">\n" +
		"  <path fill-rule=\"evenodd\" d=\"M8 2a5.53 5.53 0 0 0-3.594 1.342c-.766.66-1.321 1.52-1.464 2.383C1.266 6.095 0 7.555 0 9.318 0 11.366 1.708 13 3.781 13h8.906C14.502 13 16 11.57 16 9.773c0-1.636-1.242-2.969-2.834-3.194C12.923 3.999 10.69 2 8 2zm2.354 5.146l-2-2a.5.5 0 0 0-.708 0l-2 2a.5.5 0 1 0 .708.708L7.5 6.707V10.5a.5.5 0 0 0 1 0V6.707l1.146 1.147a.5.5 0 0 0 .708-.708z\"/>\n" +
		"</svg> Upload";
	upload_button.className = 'upload_button btn btn-danger btn-mm'
	upload_button.addEventListener("click", function(event) {
		if(uploading) {
			alert("Another upload in progress. Wait please!");
		} else {
			uploading = true
            
            sentence_id = document.getElementById("sentence_id").innerText

			var xhr = new XMLHttpRequest();
			xhr.onload = function(e) {
				if(this.readyState === 4) {
					responseText = e.target.responseText
					console.log("Server returned: ", responseText);
					var responseObj = JSON.parse(responseText);
					liNode = document.getElementById(responseObj['file_id'])
					liNode.querySelector(".loading_container").className = 'loading_container d-none'
					liNode.querySelector(".response_container").innerHTML = responseObj['message']
					if(responseObj['status'] === 'success') {
						liNode.className = liNode.className + ' list-group-item-success'
					} else {
						liNode.className = liNode.className + ' list-group-item-warning'
					}
				}

				uploading = false
			};
			var fd = new FormData();
			fd.append("audio_data", blob, unix_timestamp);
			fd.append("type_of_operation", typeOfOperation)
			fd.append("file_id", unix_timestamp)
            fd.append("sentence_id", sentence_id)
			xhr.open("POST","/upload",true);
			xhr.send(fd);

			event.target.disabled = true
			event.target.parentNode.querySelector('.loading_container').className = 'loading_container d-block'
		}
	})
	upload_button.disabled = false

	// Loading
	var loading_code  = '<div class="spinner-border" role="status">'
		loading_code += '	<span class="sr-only">Loading...</span>'
		loading_code += '</div>'
	var loading_container = document.createElement('div')
	loading_container.innerHTML = loading_code
	loading_container.className = 'loading_container d-none'

	// Add elements to <li>
	li.appendChild(date_container)
	li.appendChild(audio_element);
	var audio_container = document.createElement('div');
	audio_container.className = 'audio_container mt-3'
	//audio_container.append(download_button);
	audio_container.append(upload_button)
	audio_container.append(response_container);
	audio_container.append(loading_container)
	li.appendChild(audio_container);

	// Add <li> to <ol>
	recordingsList.appendChild(li);
}

update_sentence_button = document.getElementById("update_sentence")
update_sentence_button.addEventListener("click", function(event) {
	update_sentence_button.disabled = true
    update_sentence_button.className = 'btn btn-secondary btn-sm d-none'
    sentence_loader_element = document.getElementById("sentence_loader")
    sentence_loader_element.className = 'd-block'

	var xhr = new XMLHttpRequest();
	xhr.onload = function(e) {
        sentence_loader_element = document.getElementById("sentence_loader")
        sentence_loader_element.className = 'd-none'
        update_sentence_button.className = 'btn btn-secondary btn-sm d-block'
		if(this.readyState === 4) {
			responseText = e.target.responseText
			console.log("Server returned: ", responseText);
			var responseObj = JSON.parse(responseText);            
			sentence_element = document.getElementById("sentence")
			sentence_id_element = document.getElementById("sentence_id")
			sentence_element.innerHTML = responseObj['sentence']
			sentence_id_element.innerHTML = responseObj['sentence_id']
		}

		update_sentence_button.disabled = false
	};

	xhr.open("GET","/sentence",true);
	xhr.send();
})