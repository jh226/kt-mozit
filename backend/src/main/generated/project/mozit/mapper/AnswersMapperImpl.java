package project.mozit.mapper;

import javax.annotation.processing.Generated;
import org.springframework.stereotype.Component;
import project.mozit.domain.Answers;
import project.mozit.dto.AnswersDTO.Post;
import project.mozit.dto.AnswersDTO.Response;

@Generated(
    value = "org.mapstruct.ap.MappingProcessor",
<<<<<<< HEAD
    date = "2025-01-21T10:28:20+0900",
=======
    date = "2025-01-17T16:38:07+0900",
>>>>>>> d39e813934a412cd21484fe569c5905d24bf8fe5
    comments = "version: 1.4.1.Final, compiler: javac, environment: Java 17.0.13 (Eclipse Adoptium)"
)
@Component
public class AnswersMapperImpl implements AnswersMapper {

    @Override
    public Answers PostDTOToEntity(Post post) {
        if ( post == null ) {
            return null;
        }

        Answers answers = new Answers();

        answers.setAnswerDetail( post.getAnswerDetail() );

        return answers;
    }

    @Override
    public Response entityToResponse(Answers answer) {
        if ( answer == null ) {
            return null;
        }

        Response response = new Response();

        response.setAnswerNum( answer.getAnswerNum() );
        response.setTimestamp( answer.getTimestamp() );
        response.setAnswerDetail( answer.getAnswerDetail() );

        return response;
    }
}
