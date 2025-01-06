import { useEffect } from 'react'
import { useState } from 'react'

function App() {
  const [text,settext] = useState('')
  const [sentiment,setsentiment] = useState('')

  const get_prediction = async (e)=>{
    e.preventDefault()

    const data = {
      text
    }
    const url = "http://127.0.0.1:5000/prediction_comments"
    const options = {
      method:'POST',
      headers:{
        "Content-Type":"application/json"
      },
      body:JSON.stringify({Comments:data})
    }
    const response = await fetch(url,options)
    const response_2 = await response.json()
    setsentiment(response_2.sentiment)
  }
  return (
    <>
    <div>
      <form onSubmit = {get_prediction}>
        <h1>Sentiment Analysis AI</h1>
        <input type="text" value = {text} onChange={(e) => settext(e.target.value)}/>
        <div>
          <button type="submit">Analyze</button>
          <p>Sentiment: {sentiment}</p>
        </div>
      </form>
      
    </div>
    </>
  )
}

export default App
