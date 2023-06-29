import React, { useState, ChangeEvent } from 'react';
import { Row, Col } from 'react-bootstrap';
import { ThreeDots  } from 'react-loader-spinner'


export default function Algoritmo() {

    const [emotion, setEmotion] = useState<string>("");
    const [pictureImage, setPictureImage] = useState<string>("");
    const [loading, setLoading] = useState(false);

    const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            getEmotion(file)
            reader.addEventListener('load', (e) => {
                const readerTarget = e.target as FileReader;
                if (readerTarget) {
                    const imgDataUrl = readerTarget.result as string;
                    setPictureImage(imgDataUrl);
                    // downloadImage(imgDataUrl);
                }
            });

            reader.readAsDataURL(file);
        } else {
            setPictureImage('Escolha uma imagem');
        }
    };

    // eslint-disable-next-line
    const downloadImage = (imgDataUrl: string) => {
        const downloadLink = document.createElement('a');
        downloadLink.href = imgDataUrl;
        downloadLink.download = 'teste.jpg';

        downloadLink.click();

        downloadLink.remove();
    };



    function getEmotion(image: any) {
        setLoading(true)
        const formData = new FormData();
        formData.append('imagefile', image);
        const headers: HeadersInit = {
            'Access-Control-Allow-Origin': '*'
        }
        const requestOptions: RequestInit = {
            method: 'POST',
            headers,
            body: formData
        };
        fetch("http://localhost:5000/get-image", requestOptions)
            .then(res => res.json())
            .then(res => {
                console.log(res)
                setLoading(false)
                setEmotion(res.message)
            })
            .catch(e => {
                setLoading(false)
                setEmotion("A imagem não deu certo. Tente novamente")
                console.error(e)
            })
                
    }

    return (
        <Row className="justify-content-center">
            <Col md={6}>
                <label className="picture" htmlFor="picture__input">
                    <img alt="Insira a Imagem aqui" width="100%" height="100%" src={pictureImage} />
                </label>

                <input type="file" name="picture__input" id="picture__input"
                    onChange={handleFileChange} />
                {loading?
                <Col style={{display:'flex',justifyContent:'center'}}>
                    <ThreeDots
                    height="80" 
                    width="80" 
                    radius="9"
                    color="black" 
                    ariaLabel="three-dots-loading"
                    wrapperStyle={{}}
                    wrapperClassName=""
                    visible={true}
                    />
                </Col>
                :
                <h1 style={{backgroundColor:'black', color:'white', textAlign:'center', fontFamily:'sans-serif',  borderRadius:15 }}>{emotion}</h1>
                }
            </Col>

        </Row>


    );
};
