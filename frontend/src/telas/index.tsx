import React, { useState } from 'react';
import './style.css';

const index = () => {
  const [wikiVisible, setWikiVisible] = useState(false);
  const [devsVisible, setDevsVisible] = useState(false);

  const toggleWiki = () => {
    setWikiVisible(!wikiVisible);
    setDevsVisible(false);
  };

  const toggleDevs = () => {
    setDevsVisible(!devsVisible);
    setWikiVisible(false);
  };


  return (
    <div style={styles.html}>
      <div style={styles.body}>
        <div style={styles.container}>
          <div style={styles.inicio}>
            <span style={styles.spanIni}>
              <button
                style={styles.glowOn}
                onClick={() => (window.location.href = 'algoritmo.html')}
              >
                <input
                  type="image"
                  src="https://i.ibb.co/w7F46p7/Screenshot-5.png"
                  style={{ width: '70px', height: '60px', alignItems: 'center' }}
                />
              </button>
            </span>
          </div>
          <div className="info">
            <span className="spanInfo">
              <button style={styles.glowOn} onClick={toggleWiki}>
                Wiki
              </button>
              <span id="wiki" className={wikiVisible ? 'glow-on-hover' : 'hidden'}>
                As expressões faciais são formas de comunicação humana que transmitem diversas emoções, como alegria, tristeza, raiva, medo e surpresa. Elas são formadas pela contração dos músculos do rosto e podem ser involuntárias ou voluntárias, dependendo do contexto.
                Existem seis emoções básicas universalmente reconhecidas, mas a expressão de neutralidade também pode ser considerada uma emoção básica.
                Nos últimos anos, houve avanços na detecção de expressões faciais por meio de algoritmos e técnicas de aprendizado de máquina, como redes neurais.
                Essa tecnologia tem diversas aplicações, desde análise de bem-estar social até criação de interfaces para interação humano-computador, colaboração para diagnósticos médicos e coleta de feebacks de determinados produtos e serviços oferecidos a clientes e usuários.
                Este trabalho propõe o desenvolvimento de um algorítmo capaz de detectar, uma foto, um rosto humano e, a partir deste rosto, analisar e descobrir qual expressão facial ele está emitindo naquele momento para, assim, detectar seu respectivo sentimento.
              </span>
            </span>
            <span className="spanDevs">
              <button style={styles.glowOn} onClick={toggleDevs}>
                Devs
              </button>
              <span id="devs" className={devsVisible ? 'glow-on-hover' : 'hidden'}>
                <div style={styles.nomes}>
                  <div>
                    <img
                      src="https://i.ibb.co/SfH4r6c/mystery.jpg"
                      alt="Sem foto"
                      id="fotos"
                      style={{ float: 'left' }}
                    />
                    <p>Giovani Siervo</p>
                  </div>
                  <div>
                    <img
                      src="https://i.ibb.co/SfH4r6c/mystery.jpg"
                      alt="Sem foto"
                      id="fotos"
                      style={{ float: 'left' }}
                    />
                    <p>Gustavo Zago</p>
                  </div>
                  <div>
                    <img
                      src="https://i.ibb.co/SfH4r6c/mystery.jpg"
                      alt="Sem foto"
                      id="fotos"
                      style={{ float: 'left' }}
                    />
                    <p>Felipe Bueno</p>
                  </div>
                  <div>
                    <img
                      src="https://i.ibb.co/SfH4r6c/mystery.jpg"
                      alt="Sem foto"
                      id="fotos"
                      style={{ float: 'left' }}
                    />
                    <p>Jonathan Christofoleti</p>
                  </div>
                </div>
              </span>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

const styles = {
  html:{
    height: '100%',
    margin: '0',
  },
  body: {
    backgroundImage: "url('https://i.ibb.co/vk0shrg/dfsdfsdfdf.png')",
    height: '100%',
    margin: '0',
  },
  container: {
    display: 'flex',
    //flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    height: '100%',
  },
  inicio:{
    //user-select: none,
    marginBottom: 10,
    width: 170,
    height: 80,
  },
  spanIni:{
    width: 260,
    height: 90,
  },
  glowOn:{
    width: '100%',
    height: '100%',
    border: 'none',
    outline: 'none',
    color: '#fff',
    background: '#111',
    cursor: 'pointer',
    //position: 'relative',
    zIndex: 0,
    borderRadius: 10,
  },
  nomes:{
    display: 'flex',
    margin: 2,
    justifyItems: 'left',
    fontSize: 14,
  },
  devs:{
    display: 'none',
    borderRadius: 15,
    padding: 9,
    marginTop: 5,
    marginLeft: 10,
    width: 270,
    height: 100,
    opacity: 1,
    animationName: 'animaDev',
    animationDuration: 0.1,
  }

};

export default index;