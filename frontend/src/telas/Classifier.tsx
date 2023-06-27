import React, { useState, ChangeEvent } from 'react';

export default function Algoritmo() {
    const [pictureImage, setPictureImage] = useState('Escolha uma imagem');

    const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];

        if (file) {
            const reader = new FileReader();

            reader.addEventListener('load', (e) => {
                const readerTarget = e.target as FileReader;

                if (readerTarget) {
                    const imgDataUrl = readerTarget.result as string;
                    setPictureImage(imgDataUrl);
                    //downloadImage(imgDataUrl);
                }
            });

            reader.readAsDataURL(file);
        } else {
            setPictureImage('Escolha uma imagem');
        }
    };

    const downloadImage = (imgDataUrl: string) => {
        const downloadLink = document.createElement('a');
        downloadLink.href = imgDataUrl;
        downloadLink.download = 'teste.jpg';

        downloadLink.click();

        downloadLink.remove();
    };

    return (
        <>
            <label className="picture" htmlFor="picture__input">
                <img src={pictureImage} alt="Selecione uma Imagem" style={{width:'100%', height:'100%'}}/>
            </label>

            <input type="file" name="picture__input" id="picture__input" onChange={handleFileChange}></input>
        </>


    );
};
