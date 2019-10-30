var forestMap = {
	Otway: {
	  center: {lat: -38.792698, lng: 143.497603},
	  size: 20000
	}
};

var forests = [
	['Annya', -38.0571279, 141.4441015, 4, 'annya', 'A'],
	['Cobbob', -38.114081, 141.421034, 5, 'c','C'],
	['Hotspur', -37.936113, 141.526746, 3, 'hotspur','H'],
	['Mount clay', -38.2160934, 141.6937452, 2,'m', 'M']
];

var triangleCoords = [
		{lat: -37.359117, lng: 140.974683},
		{lat: -38.053986, lng: 140.966087},
		{lat: -38.336743, lng: 141.399824},
		{lat: -38.372258, lng: 141.366427},
		{lat: -38.400747, lng: 141.413228},
		{lat: -38.371028, lng: 141.407261},
		{lat: -38.358336, lng: 141.453876},
		{lat: -38.402724, lng: 141.527087},
		{lat: -38.428968, lng: 141.543922},
		{lat: -38.387292, lng: 141.579500},
		{lat: -38.403001, lng: 141.623304},
		{lat: -38.320024, lng: 141.597702},
		{lat: -38.263606, lng: 141.869980},
		{lat: -37.945607, lng: 141.822123},
		{lat: -37.889348, lng: 141.920502},
		{lat: -37.890158, lng: 141.729604},
		{lat: -37.720191, lng: 141.722134},
		{lat: -37.602617, lng: 141.526582},
		{lat: -37.462525, lng: 141.540498},
		{lat: -37.452011, lng: 141.482991},
		{lat: -37.377839, lng: 141.496712},
		{lat: -37.348743, lng: 141.218185},
		{lat: -37.430200, lng: 141.261922},
		{lat: -37.418963, lng: 141.165411},
		{lat: -37.448129, lng: 141.098641},
		{lat: -37.359990, lng: 140.976084}
	];

function initMap() {
	var map = new google.maps.Map(document.getElementById('map_feral'), {
		zoom: 7,
		center: {lat: -37.872583, lng: 142.642954}
	});

	for (var city in forestMap) {
		// Add the circle for this city to the map.
		var forestCircle = new google.maps.Circle({
			strokeColor: '#FF0000',
			strokeOpacity: 0.8,
			strokeWeight: 2,
			fillColor: '#FF0000',
			fillOpacity: 0.35,
			map: map,
			center: forestMap[city].center,
			radius: Math.sqrt(forestMap[city].size) * 100
		});
	}
	var bermudaTriangle = new google.maps.Polygon({
		paths: triangleCoords,
		strokeColor: '#FF0000',
		strokeOpacity: 0.8,
		strokeWeight: 2,
		fillColor: '#FF0000',
		fillOpacity: 0.15,
		map: map
	});
	setMarkers(map);
	bermudaTriangle.setMap(map);
}

function setMarkers(map) {
	var shape = {
	  coords: [1, 1, 1, 20, 18, 20, 18, 1],
	  type: 'poly'
	};
	for (var i = 0; i < forests.length; i++) {
		  var f = forests[i];
		  var marker = new google.maps.Marker({
			position: {lat: f[1], lng: f[2]},
			map: map,
			icon: '../static/images/placeholder.png',
			shape: shape,
			title: f[0],
			zIndex: f[3]
		  });
	}
}

function show_A(){
	var map = new google.maps.Map(document.getElementById('map_feral'), {
		zoom: 7,
		center: {lat: -37.872583, lng: 142.642954}
	});
	var f = forests[0];
	var marker = new google.maps.Marker({
		position: {lat: f[1], lng: f[2]},
		map: map,
		label: 'A'
	});
	var bermudaTriangle = new google.maps.Polygon({
		paths: triangleCoords,
		strokeColor: '#FF0000',
		strokeOpacity: 0.7,
		strokeWeight: 2,
		fillColor: '#FF0000',
		fillOpacity: 0.10,
		map: map
	});
	bermudaTriangle.setMap(map);
}

function show_C(){
	var map = new google.maps.Map(document.getElementById('map_feral'), {
		zoom: 7,
		center: {lat: -37.872583, lng: 142.642954}
	});
	var f = forests[1];
	var marker = new google.maps.Marker({
		position: {lat: f[1], lng: f[2]},
		map: map,
		label: 'C'
	});
	var bermudaTriangle = new google.maps.Polygon({
		paths: triangleCoords,
		strokeColor: '#FF0000',
		strokeOpacity: 0.7,
		strokeWeight: 2,
		fillColor: '#FF0000',
		fillOpacity: 0.10,
		map: map
	});
	bermudaTriangle.setMap(map);
}

function show_H(){
	var map = new google.maps.Map(document.getElementById('map_feral'), {
		zoom: 7,
		center: {lat: -37.872583, lng: 142.642954}
	});
	var f = forests[2];
	var marker = new google.maps.Marker({
		position: {lat: f[1], lng: f[2]},
		map: map,
		label: 'H'
	});
	var bermudaTriangle = new google.maps.Polygon({
		paths: triangleCoords,
		strokeColor: '#FF0000',
		strokeOpacity: 0.7,
		strokeWeight: 2,
		fillColor: '#FF0000',
		fillOpacity: 0.10,
		map: map
	});
	bermudaTriangle.setMap(map);
}

function show_M(){
	var map = new google.maps.Map(document.getElementById('map_feral'), {
		zoom: 7,
		center: {lat: -37.872583, lng: 142.642954}
	});
	var f = forests[3];
	var marker = new google.maps.Marker({
		position: {lat: f[1], lng: f[2]},
		map: map,
		label: 'M'
	});
	var bermudaTriangle = new google.maps.Polygon({
		paths: triangleCoords,
		strokeColor: '#FF0000',
		strokeOpacity: 0.7,
		strokeWeight: 2,
		fillColor: '#FF0000',
		fillOpacity: 0.10,
		map: map
	});
	bermudaTriangle.setMap(map);
}

function show_G(){
	var map = new google.maps.Map(document.getElementById('map_feral'), {
		zoom: 7,
		center: {lat: -37.872583, lng: 142.642954}
	});
	var bermudaTriangle = new google.maps.Polygon({
		paths: triangleCoords,
		strokeColor: '#FF0000',
		strokeOpacity: 0.8,
		strokeWeight: 2,
		fillColor: '#FF0000',
		fillOpacity: 0.15,
		map: map
	});
	bermudaTriangle.setMap(map);
}

function show_O(){
	var map = new google.maps.Map(document.getElementById('map_feral'), {
		zoom: 7,
		center: {lat: -37.872583, lng: 142.642954}
	});

	for (var city in forestMap) {
		// Add the circle for this city to the map.
		var forestCircle = new google.maps.Circle({
			strokeColor: '#FF0000',
			strokeOpacity: 0.8,
			strokeWeight: 2,
			fillColor: '#FF0000',
			fillOpacity: 0.35,
			map: map,
			center: forestMap[city].center,
			radius: Math.sqrt(forestMap[city].size) * 100
		});
	}
}

function setPickedMarkers(map,area) {
		var shape = {
		  coords: [1, 1, 1, 20, 18, 20, 18, 1],
		  type: 'poly'
		};
		for (var i = 0; i < forests.length; i++) {
			  var f = forests[i];
			  if (area ==f[4]){
					var marker = new google.maps.Marker({
						position: {lat: f[1], lng: f[2]},
						map: map,
						icon: '../static/images/placeholder.png',
						shape: shape,
						title: f[0],
						zIndex: f[3]
					  });
			  }
		}
}

function show_all(areas){
	var map = new google.maps.Map(document.getElementById('map_feral'), {
		zoom: 7,
		center: {lat: -37.872583, lng: 142.642954}
	});
	setPickedMarkers(map,areas );
	if (areas.includes("annya") || areas.includes("c")|| areas.includes("hotspur")|| areas.includes("m")){
		var bermudaTriangle = new google.maps.Polygon({
			paths: triangleCoords,
			strokeColor: '#FF0000',
			strokeOpacity: 0.8,
			strokeWeight: 2,
			fillColor: '#FF0000',
			fillOpacity: 0.15,
			map: map
		});
		bermudaTriangle.setMap(map);
	}

	if(areas.includes('otway')){
		for (var city in forestMap) {
			// Add the circle for this city to the map.
			var forestCircle = new google.maps.Circle({
				strokeColor: '#FF0000',
				strokeOpacity: 0.8,
				strokeWeight: 2,
				fillColor: '#FF0000',
				fillOpacity: 0.35,
				map: map,
				center: forestMap[city].center,
				radius: Math.sqrt(forestMap[city].size) * 100
			});
		}
	}
}

