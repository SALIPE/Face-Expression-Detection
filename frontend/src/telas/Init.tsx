import React, { useState } from 'react';
import { Row, Col, Card } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import fotoJonathan from '../images/jonathan.png'
import fotoFelipe from '../images/felipe.png'
import fotoGustavo from '../images/gustavo.png'
import fotoGiovanni from '../images/giovanni.png'
 

export default function Init() {
  const [wikiVisible, setWikiVisible] = useState(false);
  const [devsVisible, setDevsVisible] = useState(false);

  const navigate = useNavigate();

  const toggleWiki = () => {
    setWikiVisible(!wikiVisible);
    setDevsVisible(false);
  };

  const toggleDevs = () => {
    setDevsVisible(!devsVisible);
    setWikiVisible(false);
  };


  return (
    <Row className='justify-content-center'>
      <Col sm={5}>
        <Row className='mb-2'>
          <Col>
            <button style={{ height: 150 }}
              onClick={() => navigate("/classifier")}
              className="glow-on-hover">
              <input type="image" src="https://i.ibb.co/w7F46p7/Screenshot-5.png"
                alt='algumacoisa'
                style={{
                  width: 70,
                  height: 60,
                  alignItems: 'center'
                }} />
            </button>
          </Col>
        </Row>
        <Row >
          <Col sm={8}>
            <button style={{ height: 50 }} className="glow-on-hover" onClick={toggleWiki}>Wiki</button>
            {wikiVisible &&
              <div
                style={{
                  marginTop: 10,
                  height: 300,
                  overflow: "auto"
                }}>
                <Card>
                  <Card.Body style={{backgroundColor:'black'}}>
                    <p style={{color:'white', textAlign:'justify'}}>
                      As expressões faciais são formas de comunicação humana que transmitem diversas emoções, como alegria, tristeza, raiva, medo e surpresa. Elas são formadas pela contração dos músculos do rosto e podem ser involuntárias ou voluntárias, dependendo do contexto.
                      Existem seis emoções básicas universalmente reconhecidas, mas a expressão de neutralidade também pode ser considerada uma emoção básica.
                      Nos últimos anos, houve avanços na detecção de expressões faciais por meio de algoritmos e técnicas de aprendizado de máquina, como redes neurais.
                      Essa tecnologia tem diversas aplicações, desde análise de bem-estar social até criação de interfaces para interação humano-computador, colaboração para diagnósticos médicos e coleta de feebacks de determinados produtos e serviços oferecidos a clientes e usuários.
                      Este trabalho propõe o desenvolvimento de um algorítmo capaz de detectar, uma foto, um rosto humano e, a partir deste rosto, analisar e descobrir qual expressão facial ele está emitindo naquele momento para, assim, detectar seu respectivo sentimento.
                    </p>
                  </Card.Body>
                </Card>

              </div>
            }
          </Col>
          <Col sm={4}>
            <button style={{ height: 50 }} className="glow-on-hover" onClick={toggleDevs}>Devs</button>
            {devsVisible &&
              <div style={{background:'black', borderRadius:20}}>
                <div className='d-flex'>
                  <img src={fotoGiovanni} alt="Sem foto" id="fotos" /><p style={{color:'white'}} >Giovani Siervo</p>
                </div>
                <div className='d-flex'>
                  <img src={fotoGustavo} alt="Sem foto" id="fotos" /><p style={{color:'white'}}>Gustavo Zago</p>
                </div>
                <div className='d-flex'>
                  <img src={fotoFelipe} alt="Sem foto" id="fotos" /><p style={{color:'white'}}>Felipe Bueno</p>
                </div>
                <div className='d-flex'>
                  <img src={fotoJonathan} alt="Sem foto" id="fotos" /><p style={{color:'white'}}>Jonathan Christofoleti</p>
                </div>
              </div>}

          </Col>
        </Row>
      </Col>

    </Row>
  );
}


