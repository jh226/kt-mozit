// import * as React from 'react';
// import Accordion from '@mui/material/Accordion';
// import AccordionDetails from '@mui/material/AccordionDetails';
// import AccordionSummary from '@mui/material/AccordionSummary';
// import Box from '@mui/material/Box';
// import Container from '@mui/material/Container';
// import Link from '@mui/material/Link';
// import Typography from '@mui/material/Typography';
// import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

// export default function FAQ() {
//   const [expanded, setExpanded] = React.useState([]);

//   const handleChange = (panel) => (event, isExpanded) => {
//     setExpanded(
//       isExpanded ? [...expanded, panel] : expanded.filter((item) => item !== panel),
//     );
//   };

//   return (
//     <Box 
//         sx={{
//             display: 'flex',
//             flexDirection: 'column',  // 수직 방향으로 배치
//             alignItems: 'center',
//             justifyContent: 'flex-start', // 상단 정렬
//             minHeight: 'calc(100vh - 64px)', // AppBar를 제외한 전체 높이
//             padding: 4,
//             marginTop: '64px', // AppBar를 위한 상단 여백
//         }}
//     >
//       <Box sx={{
//           maxWidth: 1000,
//           width: '100%',
//           }}
//       >
//         <Typography variant="h4" gutterBottom sx={{ marginBottom: 2 }}>
//           Frequently Asked Questions
//         </Typography>
//         <Box sx={{ width: '100%' }}>
//           <Accordion
//             expanded={expanded.includes('panel1')}
//             onChange={handleChange('panel1')}
//           >
//             <AccordionSummary
//               expandIcon={<ExpandMoreIcon />}
//               aria-controls="panel1d-content"
//               id="panel1d-header"
//             >
//               <Typography component="span" variant="subtitle2">
//                 How do I contact customer support if I have a question or issue?
//               </Typography>
//             </AccordionSummary>
//             <AccordionDetails>
//               <Typography
//                 variant="body2"
//                 gutterBottom
//                 sx={{ maxWidth: { sm: '100%', md: '70%' } }}
//               >
//                 You can reach our customer support team by emailing&nbsp;
//                 <Link href="mailto:support@email.com">support@email.com</Link>
//                 &nbsp;or calling our toll-free number. We&apos;re here to assist you
//                 promptly.
//               </Typography>
//             </AccordionDetails>
//           </Accordion>
//           <Accordion
//             expanded={expanded.includes('panel2')}
//             onChange={handleChange('panel2')}
//           >
//             <AccordionSummary
//               expandIcon={<ExpandMoreIcon />}
//               aria-controls="panel2d-content"
//               id="panel2d-header"
//             >
//               <Typography component="span" variant="subtitle2">
//                 Can I return the product if it doesn&apos;t meet my expectations?
//               </Typography>
//             </AccordionSummary>
//             <AccordionDetails>
//               <Typography
//                 variant="body2"
//                 gutterBottom
//                 sx={{ maxWidth: { sm: '100%', md: '70%' } }}
//               >
//                 Absolutely! We offer a hassle-free return policy. If you&apos;re not
//                 completely satisfied, you can return the product within [number of
//                 days] days for a full refund or exchange.
//               </Typography>
//             </AccordionDetails>
//           </Accordion>
//           <Accordion
//             expanded={expanded.includes('panel3')}
//             onChange={handleChange('panel3')}
//           >
//             <AccordionSummary
//               expandIcon={<ExpandMoreIcon />}
//               aria-controls="panel3d-content"
//               id="panel3d-header"
//             >
//               <Typography component="span" variant="subtitle2">
//                 What makes your product stand out from others in the market?
//               </Typography>
//             </AccordionSummary>
//             <AccordionDetails>
//               <Typography
//                 variant="body2"
//                 gutterBottom
//                 sx={{ maxWidth: { sm: '100%', md: '70%' } }}
//               >
//                 Our product distinguishes itself through its adaptability, durability,
//                 and innovative features. We prioritize user satisfaction and
//                 continually strive to exceed expectations in every aspect.
//               </Typography>
//             </AccordionDetails>
//           </Accordion>
//           <Accordion
//             expanded={expanded.includes('panel4')}
//             onChange={handleChange('panel4')}
//           >
//             <AccordionSummary
//               expandIcon={<ExpandMoreIcon />}
//               aria-controls="panel4d-content"
//               id="panel4d-header"
//             >
//               <Typography component="span" variant="subtitle2">
//                 Is there a warranty on the product, and what does it cover?
//               </Typography>
//             </AccordionSummary>
//             <AccordionDetails>
//               <Typography
//                 variant="body2"
//                 gutterBottom
//                 sx={{ maxWidth: { sm: '100%', md: '70%' } }}
//               >
//                 Yes, our product comes with a [length of warranty] warranty. It covers
//                 defects in materials and workmanship. If you encounter any issues
//                 covered by the warranty, please contact our customer support for
//                 assistance.
//               </Typography>
//             </AccordionDetails>
//           </Accordion>
//         </Box>
//       </Box>
//     </Box>
//   );
// }
// import * as React from 'react';
// import Accordion from '@mui/material/Accordion';
// import AccordionDetails from '@mui/material/AccordionDetails';
// import AccordionSummary from '@mui/material/AccordionSummary';
// import Box from '@mui/material/Box';
// import Container from '@mui/material/Container';
// import Link from '@mui/material/Link';
// import Typography from '@mui/material/Typography';
// import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
// import Button from '@mui/material/Button';

// export default function FAQ() {
//   const [expanded, setExpanded] = React.useState([]);

//   // FAQ 데이터
//   const faqData = [
//     {
//       question: 'Q: 이 서비스는 어떤 목적으로 사용되나요?',
//       answer: 'A: 동영상 내 개인정보(얼굴, 차량 번호판 등)와 유해 요소를 자동으로 감지하고 모자이크 처리하는 기능을 제공합니다.',
//     },
//     {
//       question: 'Q: 어떤 유형의 파일을 지원하나요?',
//       answer: 'A: 지원되는 파일 형식은 MP4, MOV, AVI, MKV 등입니다. 파일 크기와 해상도 제한은 여기에서 확인할 수 있습니다.',
//     },
//     {
//       question: 'Q: 모자이크 처리는 어떻게 이루어지나요요?',
//       answer: 'A: AI 기반 객체 인식 기술을 사용해 얼굴, 번호판, 유해 요소 등을 자동으로 탐지한 후 모자이크를 적용합니다.',
//     },
//     {
//       question: 'Q: 동영상 모자이크를 적용하는 방법은 무엇인가요?',
//       answer: 'A: 동영상 작업 페이지에서 파일을 업로드하고 원하는 모자이크 설정을 선택한 후 처리 버튼을 누르면 됩니다.',
//     },
//     {
//       question: 'Q: 서비스는 무료인가요?',
//       answer: 'A: 플랜 구독 서비스를 통해 이용할 수 있으며, 사용자에게 맞는 플랜을 선택할 수 있습니다.',
//     },
//     {
//       question: 'Q: 내 동영상이 다른 곳에 사용될 가능성이 있나요?',
//       answer: 'A: 동영상은 어떠한 방식으로도 외부에 공유되지 않습니다.',
//     },
//     {
//       question: 'Q: 지원되는 브라우저는 무엇인가요?',
//       answer: 'A: Chrome, FireFox, Safari, Edge 등 최신 브라우저를 권장합니다.',
//     },
//     {
//       question: 'Q: 동영상 처리 속도가 느린 이유는 무엇인가요?',
//       answer: 'A: 파일 크기, 해상도, 인터넷 속도에 따라 처리 시간이 달라질 수 있습니다.',
//     },
//   ];

//   const handleChange = (panel) => (event, isExpanded) => {
//     setExpanded(isExpanded ? [...expanded, panel] : expanded.filter((item) => item !== panel));
//   };

//   return (
//     <Box
//       sx={{
//         display: 'flex',
//         flexDirection: 'column',
//         alignItems: 'center',
//         justifyContent: 'flex-start',
//         minHeight: 'calc(100vh - 64px)', // AppBar를 제외한 전체 높이
//         padding: 4,
//         marginTop: '64px', // AppBar를 위한 상단 여백
//       }}
//     >
//       <Box sx={{ maxWidth: 1000, width: '100%' }}>
//         <Typography variant="h4" gutterBottom sx={{ marginBottom: 2 }}>
//           Frequently Asked Questions
//         </Typography>

//         {/* FAQ 목록 표시 */}
//         <Box sx={{ width: '100%' }}>
//           {faqData.map((faq, index) => (
//             <Accordion key={index} expanded={expanded.includes(`panel${index}`)} onChange={handleChange(`panel${index}`)}>
//               <AccordionSummary expandIcon={<ExpandMoreIcon />} aria-controls={`panel${index}-content`} id={`panel${index}-header`}>
//                 <Typography component="span" variant="subtitle2">
//                   {faq.question}
//                 </Typography>
//               </AccordionSummary>
//               <AccordionDetails>
//                 <Typography variant="body2" gutterBottom sx={{ maxWidth: { sm: '100%', md: '70%' } }}>
//                   {faq.answer}
//                 </Typography>
//               </AccordionDetails>
//             </Accordion>
//           ))}
//         </Box>
//       </Box>
//     </Box>
//   );
// }
import * as React from 'react';
import Accordion from '@mui/material/Accordion';
import AccordionDetails from '@mui/material/AccordionDetails';
import AccordionSummary from '@mui/material/AccordionSummary';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';


export default function FAQ() {
  const [expanded, setExpanded] = React.useState([]);

  // FAQ 데이터
  const faqData = [
    {
      question: 'Q: 이 서비스는 어떤 목적으로 사용되나요?',
      answer: 'A: 동영상 내 개인정보(얼굴, 차량 번호판 등)와 유해 요소를 자동으로 감지하고 모자이크 처리하는 기능을 제공합니다.',
    },
    {
      question: 'Q: 어떤 유형의 파일을 지원하나요?',
      answer: 'A: 지원되는 파일 형식은 MP4, MOV, AVI, MKV 등입니다. 파일 크기와 해상도 제한은 여기에서 확인할 수 있습니다.',
    },
    {
      question: 'Q: 모자이크 처리는 어떻게 이루어지나요?',
      answer: 'A: AI 기반 객체 인식 기술을 사용해 얼굴, 번호판, 유해 요소 등을 자동으로 탐지한 후 모자이크를 적용합니다.',
    },
    {
      question: 'Q: 동영상 모자이크를 적용하는 방법은 무엇인가요?',
      answer: 'A: 동영상 작업 페이지에서 파일을 업로드하고 원하는 모자이크 설정을 선택한 후 처리 버튼을 누르면 됩니다.',
    },
    {
      question: 'Q: 서비스는 무료인가요?',
      answer: 'A: 플랜 구독 서비스를 통해 이용할 수 있으며, 사용자에게 맞는 플랜을 선택할 수 있습니다.',
    },
    {
      question: 'Q: 내 동영상이 다른 곳에 사용될 가능성이 있나요?',
      answer: 'A: 동영상은 어떠한 방식으로도 외부에 공유되지 않습니다.',
    },
    {
      question: 'Q: 지원되는 브라우저는 무엇인가요?',
      answer: 'A: Chrome, FireFox, Safari, Edge 등 최신 브라우저를 권장합니다.',
    },
    {
      question: 'Q: 동영상 처리 속도가 느린 이유는 무엇인가요?',
      answer: 'A: 파일 크기, 해상도, 인터넷 속도에 따라 처리 시간이 달라질 수 있습니다.',
    },
  ];

  const handleChange = (panel) => (event, isExpanded) => {
    setExpanded(isExpanded ? [...expanded, panel] : expanded.filter((item) => item !== panel));
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'flex-start',
        minHeight: 'calc(100vh - 64px)', // AppBar를 제외한 전체 높이
        padding: 4,
        marginTop: '64px', // AppBar를 위한 상단 여백
      }}
    >
      <Box sx={{ maxWidth: 900, width: '100%' }}>
        <Typography variant="h4" gutterBottom sx={{ marginBottom: 3, fontWeight: 'bold', color: '#' }}>
          Frequently Asked Questions
        </Typography>

        {/* FAQ 목록 표시 */}
        <Box sx={{ width: '100%' }}>
          {faqData.map((faq, index) => (
            <Accordion
              key={index}
              expanded={expanded.includes(`panel${index}`)}
              onChange={handleChange(`panel${index}`)}
              sx={{
                marginBottom: 2,
                borderRadius: 2,
                boxShadow: 3,
                '&:hover': {
                  boxShadow: 6, // 마우스 오버 시 그림자 강하게
                },
              }}
            >
              <AccordionSummary
                expandIcon={<ExpandMoreIcon />}
                aria-controls={`panel${index}-content`}
                id={`panel${index}-header`}
                sx={{
                  borderRadius: '8px',
                  '& .MuiAccordionSummary-content': {
                    margin: 0,
                    color: '#',
                  },
                }}
              >
                <Typography component="span" variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                  {faq.question}
                </Typography>
              </AccordionSummary>
              <AccordionDetails sx={{ borderRadius: '8px' }}>
                <Typography variant="body2" gutterBottom sx={{ color: '#' }}>
                  {faq.answer}
                </Typography>
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>
      </Box>
    </Box>
  );
}
