$(function(){
  /*$("#app").on("click", ".message-text", function(obj){
    console.log(this.textContent)
  });
  */
  setTimeout(scrape, 5000)
});

/*  Simply prints yaks on the current page, replies, 
    and images are not taken into acount. Score and 
    Text are available
*/
function scrape(){
  obj = $(".message-container")
  for(var i =0; i < obj.length; i++){
    m = obj[i]
    if ($(m).hasClass("post-message-container"))
      continue
    //score = m.children[1].innerText.trim()
    //text = m.children[2].innerText
    score = $(m).find(".likes").text().trim()
    text = $(m).find(".message-text").text()
    replies = $(m).find(".comment-count")[0].children[1]
    if(replies.children.length < 3)
      replies = 0 //no replies yet
    else
      replies = replies.children[2].children[1].innerText //traverse the html tree!
    timeInfo = $(m).find(".message-footer-item")[0].children[0].text
    console.log("{" + "'replies': " + replies +
                ", 'score': " + score + 
                ", 'text': " + text + 
                ", 'time': " + timeInfo +
                "}")
  }
}