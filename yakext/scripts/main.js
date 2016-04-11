$(function(){
  setTimeout(scrape, 5000)
});

/*  Simply prints yaks on the current page, images are 
    not taken into acount yet. Score text time and 
    replies are parsed and loged to the console as json
*/
function scrape(){
  obj = $(".message-container")
  for(var i =0; i < obj.length; i++){
    m = obj[i]
    if ($(m).hasClass("post-message-container"))
      continue

    id = $(m).attr("data-reactid")
    score = $(m).find(".likes").text().trim()
    text = $(m).find(".message-text").text()
    timeInfo = $(m).find(".message-footer-item")[0].children[0].text
    replies = $(m).find(".comment-count")[0].children[1]
    
    if(replies.children.length < 3)
      replies = 0 //no replies yet
    else
      replies = replies.children[2].children[1].innerText //traverse the html tree
    
    sendData(id + ":::" + replies + ":::" + score + ":::" + text + ":::" + timeInfo + "\r\n")
  }
  setTimeout(refreshPage, 1000 * 60 * 5) //refresh page every 5 min
}

function refreshPage(){
  location.reload()
}

function sendData(data){
  var req = new XMLHttpRequest()
  var url = "http://localhost:9999"

  req.open("POST", url, true)
  console.log(data)
  req.send(data)
}