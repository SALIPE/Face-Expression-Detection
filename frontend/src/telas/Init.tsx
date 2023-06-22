import React, { useState } from 'react';

export default function Init() {
  const [wikiVisible, setWikiVisible] = useState(false);
  const [devsVisible, setDevsVisible] = useState(false);

  const goToAlgoritmo=()=>{
    //navegar p outra tela
  }

  const toggleWiki = () => {
    setWikiVisible(!wikiVisible);
    setDevsVisible(false);
  };

  const toggleDevs = () => {
    setDevsVisible(!devsVisible);
    setWikiVisible(false);
  };


  return (
    <>
      <div className="inicio">
        <span className="spanIni">
          <button className="glow-on-hover" onClick={goToAlgoritmo}>
            <input type="image" src="https://i.ibb.co/w7F46p7/Screenshot-5.png"
              alt='algumacoisa'
              style={{
                width: 70,
                height: 60,
                alignItems: 'center'
              }} />
          </button>
        </span>
      </div>
      <div className="info">
        <span className="spanInfo">
          <button className="glow-on-hover" onClick={toggleWiki}>Wiki</button>
          {wikiVisible ? 
            <span id="wiki" className="glow-on-hover">
              As expressões faciais são formas de comunicação humana que transmitem diversas emoções, como alegria, tristeza, raiva, medo e surpresa. Elas são formadas pela contração dos músculos do rosto e podem ser involuntárias ou voluntárias, dependendo do contexto.
              Existem seis emoções básicas universalmente reconhecidas, mas a expressão de neutralidade também pode ser considerada uma emoção básica.
              Nos últimos anos, houve avanços na detecção de expressões faciais por meio de algoritmos e técnicas de aprendizado de máquina, como redes neurais.
              Essa tecnologia tem diversas aplicações, desde análise de bem-estar social até criação de interfaces para interação humano-computador, colaboração para diagnósticos médicos e coleta de feebacks de determinados produtos e serviços oferecidos a clientes e usuários.
              Este trabalho propõe o desenvolvimento de um algorítmo capaz de detectar, uma foto, um rosto humano e, a partir deste rosto, analisar e descobrir qual expressão facial ele está emitindo naquele momento para, assim, detectar seu respectivo sentimento.
            </span>
          : null
          }
        </span>
        <span className="spanInfo">
          <button className="glow-on-hover" onClick={toggleDevs}>Devs</button>
          <span id="devs" className="glow-on-hover">
            <div>
              <div>
                <img src="https://i.ibb.co/SfH4r6c/mystery.jpg" alt="Sem foto" id="fotos" style={{ float: 'left' }} /><p>Giovani Siervo</p>
              </div>
              <div>
                <img src="https://i.ibb.co/SfH4r6c/mystery.jpg" alt="Sem foto" id="fotos" style={{ float: 'left' }} /><p>Gustavo Zago</p>
              </div>
              <div>
                <img src="https://i.ibb.co/SfH4r6c/mystery.jpg" alt="Sem foto" id="fotos" style={{ float: 'left' }} /><p>Felipe Bueno</p>
              </div>
              <div>
                <img src="https://i.ibb.co/SfH4r6c/mystery.jpg" alt="Sem foto" id="fotos" style={{ float: 'left' }} /><p>Jonathan Christofoleti</p>
              </div>
            </div>
          </span>
        </span>
      </div>
    </>
  );
}


