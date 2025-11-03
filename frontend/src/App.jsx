import { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = 'http://127.0.0.1:8000';

function App() {
    const [galleryImages, setGalleryImages] = useState([]);
    const [searchResults, setSearchResults] = useState([]);
    const [selectedUploadFiles, setSelectedUploadFiles] = useState([]);
    const [selectedSearchFile, setSelectedSearchFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [searchImagePreview, setSearchImagePreview] = useState(null);
    const [textQuery, setTextQuery] = useState('');
    const [lastQueryVector, setLastQueryVector] = useState(null);
    const [feedbackStatus, setFeedbackStatus] = useState({}); // To track feedback for each image

    const fetchGallery = async () => {
        try {
            const response = await axios.get(`${API_URL}/gallery/`);
            setGalleryImages(response.data.images || []);
        } catch (error) {
            console.error("Error fetching gallery:", error);
        }
    };

    useEffect(() => {
        fetchGallery();
    }, []);

    const handleUploadFileChange = (e) => {
        setSelectedUploadFiles([...e.target.files]);
    };

    const handleSearchFileChange = (e) => {
        const file = e.target.files[0];
        setSelectedSearchFile(file);

        if (searchImagePreview) {
            URL.revokeObjectURL(searchImagePreview);
        }

        if (file) {
            setSearchImagePreview(URL.createObjectURL(file));
        } else {
            setSearchImagePreview(null);
        }
    };

    const handleUpload = async () => {
        if (selectedUploadFiles.length === 0) return;
        const formData = new FormData();
        for (const file of selectedUploadFiles) {
            formData.append("files", file);
        }

        setLoading(true);
        try {
            await axios.post(`${API_URL}/uploads/`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            alert(`${selectedUploadFiles.length}개의 이미지를 성공적으로 업로드했습니다!`);
            setSelectedUploadFiles([]);
            document.getElementById('upload-file-input').value = ''
            fetchGallery(); // Refresh gallery
        } catch (error) {
            console.error("Error uploading images:", error);
            alert('이미지 업로드 중 오류가 발생했습니다.');
        } finally {
            setLoading(false);
        }
    };

    const handleSearch = async () => {
        if (!selectedSearchFile) return;
        const formData = new FormData();
        formData.append("file", selectedSearchFile);
        if (textQuery.trim()) {
            formData.append("text_query", textQuery);
        }

        setLoading(true);
        setSearchResults([]);
        setLastQueryVector(null);
        setFeedbackStatus({});

        try {
            const response = await axios.post(`${API_URL}/search/`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            setSearchResults(response.data.results || []);
            setLastQueryVector(response.data.query_vector || null);
        } catch (error) {
            console.error("Error searching for image:", error);
            alert('이미지 검색 중 오류가 발생했습니다.');
        } finally {
            setLoading(false);
        }
    };

    const handleFeedback = async (resultFilename, judgment) => {
        if (!lastQueryVector) {
            alert("피드백을 전송할 검색 정보가 없습니다.");
            return;
        }

        setFeedbackStatus({ ...feedbackStatus, [resultFilename]: '전송중...' });

        try {
            await axios.post(`${API_URL}/feedback/`, {
                query_vector: lastQueryVector,
                result_filename: resultFilename,
                judgment: judgment
            });
            setFeedbackStatus({ ...feedbackStatus, [resultFilename]: '완료!' });
        } catch (error) {
            console.error("Error submitting feedback:", error);
            setFeedbackStatus({ ...feedbackStatus, [resultFilename]: '오류' });
            alert("피드백 전송 중 오류가 발생했습니다.");
        }
    };

    const getImageUrl = (filename) => {
        return `${API_URL}/storage/images/${filename}`;
    }

    return (
        <div className="container mt-4">
            <h1 className="mb-4 text-center">이미지 유사도 검색</h1>
            
            {loading && (
                <div className="position-fixed top-0 start-0 w-100 h-100 d-flex justify-content-center align-items-center" style={{ backgroundColor: 'rgba(0,0,0,0.5)', zIndex: 9999 }}>
                    <div className="spinner-border text-light" role="status">
                        <span className="visually-hidden">로딩중...</span>
                    </div>
                </div>
            )}

            <div className="row">
                {/* Upload Section */}
                <div className="col-md-6">
                    <div className="card">
                        <div className="card-body">
                            <h3 className="card-title">1. 갤러리에 이미지 추가</h3>
                            <div className="mb-3">
                                <input id="upload-file-input" className="form-control" type="file" onChange={handleUploadFileChange} multiple />
                            </div>
                            <button className="btn btn-primary" onClick={handleUpload} disabled={selectedUploadFiles.length === 0 || loading}>
                                {selectedUploadFiles.length > 0 ? `${selectedUploadFiles.length}개 이미지 업로드` : '이미지 업로드'}
                            </button>
                        </div>
                    </div>
                </div>

                {/* Search Section */}
                <div className="col-md-6">
                    <div className="card">
                        <div className="card-body">
                            <h3 className="card-title">2. 유사한 이미지 찾기</h3>
                            <div className="mb-3">
                                <input className="form-control" type="file" onChange={handleSearchFileChange} />
                            </div>
                            <div className="mb-3">
                                <input 
                                    className="form-control"
                                    type="text" 
                                    placeholder="부가 설명 (예: 세면대)" 
                                    value={textQuery}
                                    onChange={(e) => setTextQuery(e.target.value)}
                                />
                            </div>
                            <button className="btn btn-success" onClick={handleSearch} disabled={!selectedSearchFile || loading}>
                                검색
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Search Query & Results */}
            {searchResults.length > 0 && (
                <div className="mt-5">
                    <div className="row">
                        <div className="col-12 col-md-4 mb-4">
                            <h4>검색한 이미지</h4>
                            <div className="card">
                                <img src={searchImagePreview} className="card-img-top" alt="Query" style={{height: '300px', objectFit: 'cover'}} />
                            </div>
                        </div>
                        <div className="col-12 col-md-8">
                            <h4>검색 결과</h4>
                            <div className="row">
                                {searchResults.map((result, index) => (
                                    <div key={index} className="col-lg-4 col-md-6 col-sm-6 mb-4">
                                        <div className="card h-100">
                                            <img src={getImageUrl(result.filename)} className="card-img-top" alt={result.filename} style={{height: '300px', objectFit: 'cover'}}/>
                                            <div className="card-body">
                                                <p className="card-text">유사도: {result.similarity.toFixed(4)}</p>
                                            </div>
                                            <div className="card-footer">
                                                <small className="me-2">이 결과가 정확한가요?</small>
                                                <button 
                                                    className="btn btn-sm btn-success me-2"
                                                    onClick={() => handleFeedback(result.filename, 'Correct')}
                                                    disabled={feedbackStatus[result.filename]}
                                                >
                                                    정답
                                                </button>
                                                <button 
                                                    className="btn btn-sm btn-danger"
                                                    onClick={() => handleFeedback(result.filename, 'Incorrect')}
                                                    disabled={feedbackStatus[result.filename]}
                                                >
                                                    오답
                                                </button>
                                                {feedbackStatus[result.filename] && <small className="ms-1 text-muted">({feedbackStatus[result.filename]})</small>}
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Image Gallery */}
            <div className="mt-5">
                <h2>이미지 갤러리</h2>
                <hr />
                {galleryImages.length > 0 ? (
                    <div className="row">
                        {galleryImages.map((filename, index) => (
                            <div key={index} className="col-lg-2 col-md-3 col-sm-4 mb-4">
                                <div className="card h-100">
                                    <img src={getImageUrl(filename)} className="card-img-top" alt={filename} style={{height: '300px', objectFit: 'cover'}}/>
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <p>갤러리에 이미지가 없습니다. 이미지를 업로드하여 시작하세요!</p>
                )}
            </div>
        </div>
    );
}

export default App;