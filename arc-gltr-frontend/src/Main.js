import './App.css';
import React from 'react'
import IllinoisLOGO from './images/illinois-tech-with-seal.svg'
import Upload from './images/upload.svg'
import ZipIcon from './images/zip.png'
import PdfIcon from './images/pdf.png'

export default function Main(props) {
    const [selectedFiles, setSelectedFiles] = React.useState(undefined);
    const [loading, setLoading] = React.useState(false);
    const [complete, setComplete] = React.useState(false)
    const [msg, setMsg] = React.useState("Report Downloaded Successfully ✅")
    const uploadFiles = () => {
        upload('http://ec2-34-224-40-228.compute-1.amazonaws.com:5001/upload', selectedFiles);
    }

    async function upload(url, attachments) {
        try {
            setLoading(true);
            var formData = new FormData();
            formData.append("file", attachments[0]);
            const response = await fetch(url, {
            method: 'POST',
            body: formData
            }).then(response=>{
                response.blob().then(blob => {
                let url = window.URL.createObjectURL(blob);
                let a = document.createElement('a');
                a.href = url;
                a.download = 'output.zip';
                a.click();
                setLoading(false);
                setSelectedFiles(undefined);
                setComplete(true);
                setMsg("Report Downloaded Successfully ✅");
                var myVar = setTimeout(()=>{
                    setMsg("Something went wrong");
                    // window.location.reload();
                    setComplete(false);
                    clearTimeout(myVar);
                }, 4500);
            });
            });
            const json = await response.json();
            return json;
        } catch (err) {
            console.log(err)
            setMsg("Something went wrong");
            setComplete(true);
            var myVar = setTimeout(()=>{
                // window.location.reload();
                setComplete(false);
                clearTimeout(myVar);
            }, 4500);
        }
    }


    const selectFiles = (event) => {
        setSelectedFiles(event.target.files)
    }

    React.useEffect(() => {

    }, [])

    const getIcon = (filename) => {
        if (filename.endsWith(".zip")) {
            return <img src={ZipIcon} alt="zip" width="30px" style={{marginRight: "10px"}}/>
        } else if (filename.endsWith(".pdf")) {
            return <img src={PdfIcon} alt="pdf" width="30px" style={{marginRight: "10px"}}/>
        } else {
            return <img src={PdfIcon} alt="pdf" width="30px" style={{marginRight: "10px"}}/>
        }
    }

    return(
        <>
            <div className='container'>
                <div className='card'>
                    <h2 className='m-0'>Upload your files</h2>
                    <p  className='m-0' style={{color:"#C4C4C4", marginTop:"5px"}}>File should be txt, docx, doc, pdf, zip</p>
                    <label className="btn btn-default p-0">
                        <input style={{display:"none"}} type="file" multiple onChange={selectFiles} accept=".doc, .docx,.txt,.pdf, .zip"/>
                        <div className='card-inner' style={{fontSize:"14px"}} multiple onChange={selectFiles} accept=".doc, .docx,.txt,.pdf, .zip"  >
                            <img style={{color:"#6696F1"}} width="50px" src={Upload} alt='upload'/>
                            <p  className='m-0' style={{color:"rgb(123 123 123)", marginTop:"5px"}}>Click here to select files</p>
                        </div>
                    </label>
                    { 
                        loading && <div style={{display: "flex", justifyContent: "center"}}>
                            <p style={{color: "#C4C4C4", fontWeight: "600"}}>Uploading files..</p>
                        </div>}
                    {
                        complete && <div style={{display: "flex", justifyContent: "center"}}>
                            <p style={{color: "#000", fontWeight: "600"}}> {msg} </p>
                        </div>
                    }
                    <div>
                        {
                            selectedFiles && selectedFiles.length > 0 && !loading && (
                                <div>
                                    <p style={{color: "#C4C4C4", fontWeight: "600"}}>Selected files:</p>
                                        {Array.from(selectedFiles).map((file, index) => (
                                            <>
                                                <div style={{display: "flex", alignItems:"center",  background: "#fff", padding: "5px", marginBottom: "1px"}}>
                                                    {getIcon(file.name)}
                                                    {file.name}
                                                </div>
                                            </>
                                        ))}
                                </div>
                            )
                        }
                    </div>
                    <div>
                         <button
                            className='button'
                            style={ selectedFiles?{marginTop: "10px", }:{marginTop: "10px", background: "#C4C4C4"}}
                            disabled={!selectedFiles}
                            onClick={uploadFiles}
                            >
                            Upload
                        </button>
                    </div>
                    <div style={{paddingTop:"20px"}}>
                        <img src={IllinoisLOGO} alt="illinois tech" width="200px"/>
                    </div>
                </div>
            </div>
        </>
    );
}