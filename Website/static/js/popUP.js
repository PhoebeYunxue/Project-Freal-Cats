
var status = 0;
var xmlHttp;
function createXMLHttpRequest() {
	// 表示当前浏览器不是ie,如ns,firefox
	if (window.XMLHttpRequest) {
		xmlHttp = new XMLHttpRequest();
	} else if (window.ActiveXObject) {
		xmlHttp = new ActiveXObject("Microsoft.XMLHTTP");
	}
}
function validate(field) {
	if (field.value.trim().length != 0) {
		// 创建Ajax核心对象XMLHttpRequest
		createXMLHttpRequest();
 
		var url = '/xxx?oldpwd='
				+ field.value.trim() + '&time=' + new Date().getTime();
 
		// 设置请求方式为GET，设置请求的URL，设置为异步提交
		xmlHttp.open("GET", url, true);
 
		// 将方法地址复制给onreadystatechange属性
		// 类似于电话号码
		xmlHttp.onreadystatechange = callback;
 
		// 将设置信息发送到Ajax引擎
		xmlHttp.send(null);
	} else {
		document.getElementById("spanUserId").innerHTML = "";
	}
}
// 回调函数
function callback() {
	// Ajax引擎状态为成功
	if (xmlHttp.readyState == 4) {
		// HTTP协议状态为成功
		if (xmlHttp.status == 200) {
			if (xmlHttp.responseText == "0") {
				status = 0;
				document.getElementById("wrongpwd").innerHTML = "原密码错误";
			} else if (xmlHttp.responseText == "1") {
				status = 1;
				document.getElementById("wrongpwd").innerHTML = "";
			}
		} else {
			alert("请求失败，错误码=" + xmlHttp.status);
		}
	}
}
 
function showDiv() {
	document.getElementById('popDiv').style.display = 'block';
	document.getElementById('popIframe').style.display = 'block';
	document.getElementById('bg').style.display = 'block';
};
function closeDiv() {
	document.getElementById('popDiv').style.display = 'none';
	document.getElementById('bg').style.display = 'none';
	document.getElementById('popIframe').style.display = 'none';
	document.getElementById('oldpwd').value = '';
	document.getElementById('newpwd1').value = '';
	document.getElementById('newpwd2').value = '';
	document.getElementById("wrongpwd").innerHTML = "";
 
};
function subForm() {
	var oldPasswd = document.getElementById("oldpwd").value;
	var newPasswd = document.getElementById("newpwd1").value;
	var confirmPasswd = document.getElementById("newpwd2").value;
	alert(oldPasswd + "&" + newPasswd + "&" + confirmPasswd);
	if (oldPasswd.length == 0) {
		return false;
	}
	if (newPasswd.length == 0) {
		return false;
	}
	if (confirmPasswd.length == 0) {
		return false;
	}
 
	// 设置status是一个全局变量，0代表原密码输入错误，禁止提交表单
	if (status == 0) {
		return false;
	}
	alert("111");
	$.post("/xxx", {
		"oldPasswd" : oldPasswd,
		"newPasswd" : newPasswd,
		"confirmPasswd" : confirmPasswd
	}, function(data) {
		if (data.isResultOk) {
			alert("修改成功，请重新登录");
			window.location.href = "/xxx";
		} else {
			alert(data.resultMsg);
		}
	});
 
}