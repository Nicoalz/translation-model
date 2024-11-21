
import { useState, useCallback } from 'react'

export default function Home() {
  const [inputText, setInputText] = useState('')
  const [translatedText, setTranslatedText] = useState('')

  const translateText = useCallback(async () => {
    if (inputText.length === 0) {
      setTranslatedText('')
      return
    }

    const response = await fetch('http://127.0.0.1:8000/translate', {
      method: 'POST',
      body: JSON.stringify({ french_text: inputText }),
      headers: {
        'Content-Type': 'application/json',
      },
    })

    const { translation } = await response.json()
    setTranslatedText(translation)
  }, [inputText])

  return (
    <div className="flex flex-col min-h-screen bg-background">
      <main className="flex-grow flex items-center justify-center px-4 py-8">
        <div className="w-full max-w-4xl">
          <h1 className="text-2xl font-bold text-primary mb-6 text-center">Tr<span className='text-blue-500'>AI</span>ducteur</h1>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <div className="text-sm font-medium text-muted-foreground">Français</div>
              <div className="text-sm font-medium text-muted-foreground">Anglais</div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <textarea
                placeholder="Entrez le texte à traduire"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="h-40 resize-none border rounded-md p-2 focus:outline-none"
                aria-label="Texte à traduire"
              />
              <textarea
                value={translatedText}
                readOnly
                className="h-40 resize-none bg-muted border rounded-md p-2 focus:outline-none"
                aria-label="Traduction"
              />
            </div>
          </div>
          <button onClick={translateText} className="mt-4 w-full bg-black text-white font-medium py-2 rounded-md focus:outline-none">
            Traduire
          </button>
        </div>
      </main>
    </div>
  )
}
