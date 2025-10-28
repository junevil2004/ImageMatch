import { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = 'http://127.0.0.1:8000';

function App() {
    const [galleryImages, setGalleryImages] = useState([]);
    const [searchResults, setSearchResults] = useState([]);
    const [selectedUploadFile, setSelectedUploadFile] = useState(null);
    const [selectedSearchFile, setSelectedSearchFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [searchImagePreview, setSearchImagePreview] = useState(null);

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
        setSelectedUploadFile(e.target.files[0]);
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
        if (!selectedUploadFile) return;
        const formData = new FormData();
        formData.append("file", selectedUploadFile);
        setLoading(true);
        try {
            await axios.post(`${API_URL}/upload/`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            alert('이미지를 성공적으로 업로드했습니다!');
            setSelectedUploadFile(null);
            document.getElementById('upload-file-input').value = ''
            fetchGallery(); // Refresh gallery
        } catch (error) {
            console.error("Error uploading image:", error);
            alert('이미지 업로드 중 오류가 발생했습니다.');
        } finally {
            setLoading(false);
        }
    };

    const handleSearch = async () => {
        if (!selectedSearchFile) return;
        const formData = new FormData();
        formData.append("file", selectedSearchFile);
        setLoading(true);
        setSearchResults([]);
        try {
            const response = await axios.post(`${API_URL}/search/`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            setSearchResults(response.data.results || []);
        } catch (error) {
            console.error("Error searching for image:", error);
            alert('이미지 검색 중 오류가 발생했습니다.');
        } finally {
            setLoading(false);
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
                                <input id="upload-file-input" className="form-control" type="file" onChange={handleUploadFileChange} />
                            </div>
                            <button className="btn btn-primary" onClick={handleUpload} disabled={!selectedUploadFile || loading}>
                                이미지 업로드
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
