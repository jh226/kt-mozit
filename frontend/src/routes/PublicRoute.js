import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from '../Context/AuthContext';
import MarketingPage from '../marketing-page/MarketingPage';
import SignIn from '../sign-in/SignIn';
import SignUp from '../sign-up/SignUp';
import Agree from '../sign-up/Agree';
import FAQPage from '../FAQ-page/FAQPage';
import AboutUs from '../aboutus-page/aboutus';
import NoticePage from '../notice-page/NoticePage';
import NoticeDetailPage from '../notice-page/NoticeDetailPage';
import QuestionPage from '../question-page/QuestionPage';
import Edit from '../edit-page/EditPage';
import MyQuestionPage from '../my-page/MyQuestionPage';
import MyQuestionDetailPage from '../my-page/MyQuestionDetailPage';

import MozaicPage from '../mozaic-page/MozaicPage';
import MyPageUpdate from '../my-page/MyPageUpdate';
import MySubPage from '../my-page/MySubPage';
import MyWorkPage from '../my-page/MyWorkPage';

const PublicRoutes = () => {
    const { userId } = useAuth();
    return (
        <Routes>
        {/* Public Routes */}
        <Route path="*" element={<Navigate to="/" replace />} />
        <Route path="/" element={<MarketingPage />} />
        <Route path="/agree" element={<Agree />} />
        <Route path="/sign-in" element={<SignIn />} />
        <Route path="/signup" element={<SignUp />} />
        <Route path="/faq" element={<FAQPage />} />
        <Route path="/notice" element={<NoticePage />} />
        <Route path="/notice/:id" element={<NoticeDetailPage />} />
        <Route path="/aboutus" element={<AboutUs />} />

        {/* Private Routes */}
        {userId ? (
        <>
            <Route path="/question" element={<QuestionPage />} />
            <Route path="/edit" element={<Edit />} />
            <Route path="/myquestion" element={<MyQuestionPage />} />
            <Route path="/myquestion/:id" element={<MyQuestionDetailPage />} />
            <Route path="/mypageupdate" element={<MyPageUpdate />} />
            <Route path="/mysubpage" element={<MySubPage />} />
            <Route path="/myworkpage" element={<MyWorkPage />} />
            <Route path="/mozaic" element={<MozaicPage />} />
        </>
        ) : (
        <Route path="*" element={<Navigate to="/sign-in" replace />} />
        )}
        </Routes>
    );
};

export default PublicRoutes;
