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
                    downloadImage(imgDataUrl);
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
                <span className="picture__image"></span>
            </label>

            <input type="file" name="picture__input" id="picture__input"></input>
        </>


    );
};

const styles = {
    picture: {
        //userSelect: 'none',
        width: 600,
        aspectRatio: 16 / 9,
        borderRadius: 15,
        marginTop: '13%',
        marginLeft: '28%',
        background: '#ddd',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: '#aaa',
        //border: 2 dashed currentcolor,
        cursor: 'pointer',
        fontFamily: 'sans-serif',
        //transition: color 300ms ease-in-out, background 300ms ease-in-out,
        outline: 'none',
        overflow: 'hidden',
    },
    body: {
        backgroundImage: "url('https://i.ibb.co/vk0shrg/dfsdfsdfdf.png')",
        height: '100%',
        with: '100%',
        margin: '0',
    },
};
