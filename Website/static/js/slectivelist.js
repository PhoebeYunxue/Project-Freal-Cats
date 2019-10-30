function choose_service()
{
    var servicelist = document.getElementById("servicelist");
    servicelist.style.left = (document.body.clientWidth-450)/2+"px";
    servicelist.style.display = "block";
}
function chk_choose()
{
    document.getElementById("select").value = "";
    document.getElementById("choose_service").innerHTML = "";
    var servicelist = document.getElementById("servicelist");
    var chks = document.getElementsByTagName("input");
    var vals = "";
    var names = "";
    for(var i = 0;i<chks.length;i++)
    {
        var chk = chks[i];
        if(chk.type!="checkbox"&&chk.type!="CHECKBOX")
        continue;
        if(chk.id.indexOf("chk_service_")==-1)
        continue;
        var user_id = chk.id.replace("chk_service_","");
        if(chk.checked){
            vals+=chk.value+",";
            names+=document.getElementById("label_service_"+user_id).innerHTML+",";
        }
    }
    names = names.substring(0,names.length-1);
    vals = vals.substring(0,vals.length-1);

    document.getElementById("select").value = vals?vals:"";
    document.getElementById("choose_service").innerHTML = names?names:"请选择";
}