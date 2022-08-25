/*
* MIT License
*
* Copyright(c) 2018 Jimmie Bergmann
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files(the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions :
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
*/

#include "Yaml.hpp"
#include <memory>
#include <fstream>
#include <sstream>
#include <vector>
#include <list>
#include <cstdio>
#include <stdarg.h>


// Implementation access definitions.
#define NODE_IMP static_cast<NodeImp*>(m_pImp)
#define NODE_IMP_EXT(node) static_cast<NodeImp*>(node.m_pImp)
#define TYPE_IMP static_cast<NodeImp*>(m_pImp)->m_pImp


#define IT_IMP static_cast<IteratorImp*>(m_pImp)


namespace Yaml
{
    class ReaderLine;

    // Exception message definitions.
    static const std::string g_ErrorInvalidCharacter        = "Invalid character found.";
    static const std::string g_ErrorKeyMissing              = "Missing key.";
    static const std::string g_ErrorKeyIncorrect            = "Incorrect key.";
    static const std::string g_ErrorValueIncorrect          = "Incorrect value.";
    static const std::string g_ErrorTabInOffset             = "Tab found in offset.";
    static const std::string g_ErrorBlockSequenceNotAllowed = "Sequence entries are not allowed in this context.";
    static const std::string g_ErrorUnexpectedDocumentEnd   = "Unexpected document end.";
    static const std::string g_ErrorDiffEntryNotAllowed     = "Different entry is not allowed in this context.";
    static const std::string g_ErrorIncorrectOffset         = "Incorrect offset.";
    static const std::string g_ErrorSequenceError           = "Error in sequence node.";
    static const std::string g_ErrorCannotOpenFile          = "Cannot open file.";
    static const std::string g_ErrorIndentation             = "Space indentation is less than 2.";
    static const std::string g_ErrorInvalidBlockScalar      = "Invalid block scalar.";
    static const std::string g_ErrorInvalidQuote            = "Invalid quote.";
    static const std::string g_EmptyString                  = "";
    static Yaml::Node        g_NoneNode;

    // Global function definitions. Implemented at end of this source file.
    static std::string ExceptionMessage(const std::string & message, ReaderLine & line);
    static std::string ExceptionMessage(const std::string & message, ReaderLine & line, const size_t errorPos);
    static std::string ExceptionMessage(const std::string & message, const size_t errorLine, const size_t errorPos);
    static std::string ExceptionMessage(const std::string & message, const size_t errorLine, const std::string & data);

    static bool FindQuote(const std::string & input, size_t & start, size_t & end, size_t searchPos = 0);
    static size_t FindNotCited(const std::string & input, char token, size_t & preQuoteCount);
    static size_t FindNotCited(const std::string & input, char token);
    static bool ValidateQuote(const std::string & input);
    static void CopyNode(const Node & from, Node & to);
    static bool ShouldBeCited(const std::string & key);
    static void AddEscapeTokens(std::string & input, const std::string & tokens);
    static void RemoveAllEscapeTokens(std::string & input);

    // Exception implementations
    Exception::Exception(const std::string & message, const eType type) :
        std::runtime_error(message),
        m_Type(type)
    {
    }

    Exception::eType Exception::Type() const
    {
        return m_Type;
    }

    const char * Exception::Message() const
    {
        return what();
    }

    InternalException::InternalException(const std::string & message) :
        Exception(message, InternalError)
    {

    }

    ParsingException::ParsingException(const std::string & message) :
        Exception(message, ParsingError)
    {

    }

    OperationException::OperationException(const std::string & message) :
        Exception(message, OperationError)
    {

    }


    class TypeImp
    {

    public:

        virtual ~TypeImp()
        {
        }

        virtual const std::string & GetData() const = 0;
        virtual bool SetData(const std::string & data) = 0;
        virtual size_t GetSize() const = 0;
        virtual Node * GetNode(const size_t index) = 0;
        virtual Node * GetNode(const std::string & key) = 0;
        virtual Node * Insert(const size_t index) = 0;
        virtual Node * PushFront() = 0;
        virtual Node * PushBack() = 0;
        virtual void Erase(const size_t index) = 0;
        virtual void Erase(const std::string & key) = 0;

    };

    class SequenceImp : public TypeImp
    {

    public:

        ~SequenceImp()
        {
            for(auto it = m_Sequence.begin(); it != m_Sequence.end(); it++)
            {
                delete it->second;
            }
        }

        virtual const std::string & GetData() const
        {
            return g_EmptyString;
        }

        virtual bool SetData(const std::string & data)
        {
            return false;
        }

        virtual size_t GetSize() const
        {
            return m_Sequence.size();
        }

        virtual Node * GetNode(const size_t index)
        {
            auto it = m_Sequence.find(index);
            if(it != m_Sequence.end())
            {
                return it->second;
            }
            return nullptr;
        }

        virtual Node * GetNode(const std::string & key)
        {
            return nullptr;
        }

        virtual Node * Insert(const size_t index)
        {
            if(m_Sequence.size() == 0)
            {
                Node * pNode = new Node;
                m_Sequence.insert({0, pNode});
                return pNode;
            }

            if(index >= m_Sequence.size())
            {
                auto it = m_Sequence.end();
                --it;
                Node * pNode = new Node;
                m_Sequence.insert({it->first, pNode});
                return pNode;
            }

            auto it = m_Sequence.cbegin();
            while(it != m_Sequence.cend())
            {
                m_Sequence[it->first+1] = it->second;

                if(it->first == index)
                {
                    break;
                }
            }

            Node * pNode = new Node;
            m_Sequence.insert({index, pNode});
            return pNode;
        }

        virtual Node * PushFront()
        {
            for(auto it = m_Sequence.cbegin(); it != m_Sequence.cend(); it++)
            {
                m_Sequence[it->first+1] = it->second;
            }

            Node * pNode = new Node;
            m_Sequence.insert({0, pNode});
            return pNode;
        }

        virtual Node * PushBack()
        {
            size_t index = 0;
            if(m_Sequence.size())
            {
                auto it = m_Sequence.end();
                --it;
                index = it->first + 1;
            }

            Node * pNode = new Node;
            m_Sequence.insert({index, pNode});
            return pNode;
        }

        virtual void Erase(const size_t index)
        {
            auto it = m_Sequence.find(index);
            if(it == m_Sequence.end())
            {
                return;
            }
            delete it->second;
            m_Sequence.erase(index);
        }

        virtual void Erase(const std::string & key)
        {
        }

        std::map<size_t, Node*> m_Sequence;

    };

    class MapImp : public TypeImp
    {

    public:

        ~MapImp()
        {
            for(auto it = m_Map.begin(); it != m_Map.end(); it++)
            {
                delete it->second;
            }
        }

        virtual const std::string & GetData() const
        {
            return g_EmptyString;
        }

        virtual bool SetData(const std::string & data)
        {
            return false;
        }

        virtual size_t GetSize() const
        {
            return m_Map.size();
        }

        virtual Node * GetNode(const size_t index)
        {
            return nullptr;
        }

        virtual Node * GetNode(const std::string & key)
        {
            auto it = m_Map.find(key);
            if(it == m_Map.end())
            {
                Node * pNode = new Node;
                m_Map.insert({key, pNode});
                return pNode;
            }
            return it->second;
        }

        virtual Node * Insert(const size_t index)
        {
            return nullptr;
        }

        virtual Node * PushFront()
        {
            return nullptr;
        }

        virtual Node * PushBack()
        {
            return nullptr;
        }

        virtual void Erase(const size_t index)
        {
        }

        virtual void Erase(const std::string & key)
        {
            auto it = m_Map.find(key);
            if(it == m_Map.end())
            {
                return;
            }
            delete it->second;
            m_Map.erase(key);
        }

        std::map<std::string, Node*> m_Map;

    };

    class ScalarImp : public TypeImp
    {

    public:

        ~ScalarImp()
        {
        }

        virtual const std::string & GetData() const
        {
            return m_Value;
        }

        virtual bool SetData(const std::string & data)
        {
            m_Value = data;
            return true;
        }

        virtual size_t GetSize() const
        {
            return 0;
        }

        virtual Node * GetNode(const size_t index)
        {
            return nullptr;
        }

        virtual Node * GetNode(const std::string & key)
        {
            return nullptr;
        }

        virtual Node * Insert(const size_t index)
        {
            return nullptr;
        }

        virtual Node * PushFront()
        {
            return nullptr;
        }

        virtual Node * PushBack()
        {
            return nullptr;
        }

        virtual void Erase(const size_t index)
        {
        }

        virtual void Erase(const std::string & key)
        {
        }

        std::string m_Value;

    };


    // Node implementations.
    class NodeImp
    {

    public:

        NodeImp() :
            m_Type(Node::None),
            m_pImp(nullptr)
        {
        }

        ~NodeImp()
        {
            Clear();
        }

        void Clear()
        {
            if(m_pImp != nullptr)
            {
                delete m_pImp;
                m_pImp = nullptr;
            }
            m_Type = Node::None;
        }

        void InitSequence()
        {
            if(m_Type != Node::SequenceType || m_pImp == nullptr)
            {
                if(m_pImp)
                {
                    delete m_pImp;
                }
                m_pImp = new SequenceImp;
                m_Type = Node::SequenceType;
            }
        }

        void InitMap()
        {
            if(m_Type != Node::MapType || m_pImp == nullptr)
            {
                if(m_pImp)
                {
                    delete m_pImp;
                }
                m_pImp = new MapImp;
                m_Type = Node::MapType;
            }
        }

        void InitScalar()
        {
            if(m_Type != Node::ScalarType || m_pImp == nullptr)
            {
                if(m_pImp)
                {
                    delete m_pImp;
                }
                m_pImp = new ScalarImp;
                m_Type = Node::ScalarType;
            }

        }

        Node::eType    m_Type;  ///< Type of node.
        TypeImp *      m_pImp;  ///< Imp of type.

    };

    // Iterator implementation class
    class IteratorImp
    {

    public:

        virtual ~IteratorImp()
        {
        }

        virtual Node::eType GetType() const = 0;
        virtual void InitBegin(SequenceImp * pSequenceImp) = 0;
        virtual void InitEnd(SequenceImp * pSequenceImp) = 0;
        virtual void InitBegin(MapImp * pMapImp) = 0;
        virtual void InitEnd(MapImp * pMapImp) = 0;

    };

    class SequenceIteratorImp : public IteratorImp
    {

    public:

        virtual Node::eType GetType() const
        {
            return Node::SequenceType;
        }

        virtual void InitBegin(SequenceImp * pSequenceImp)
        {
            m_Iterator = pSequenceImp->m_Sequence.begin();
        }

        virtual void InitEnd(SequenceImp * pSequenceImp)
        {
            m_Iterator = pSequenceImp->m_Sequence.end();
        }

        virtual void InitBegin(MapImp * pMapImp)
        {
        }

        virtual void InitEnd(MapImp * pMapImp)
        {
        }

        void Copy(const SequenceIteratorImp & it)
        {
            m_Iterator = it.m_Iterator;
        }

        std::map<size_t, Node *>::iterator m_Iterator;

    };

    class MapIteratorImp : public IteratorImp
    {

    public:

        virtual Node::eType GetType() const
        {
            return Node::MapType;
        }

        virtual void InitBegin(SequenceImp * pSequenceImp)
        {
        }

        virtual void InitEnd(SequenceImp * pSequenceImp)
        {
        }

        virtual void InitBegin(MapImp * pMapImp)
        {
            m_Iterator = pMapImp->m_Map.begin();
        }

        virtual void InitEnd(MapImp * pMapImp)
        {
            m_Iterator = pMapImp->m_Map.end();
        }

        void Copy(const MapIteratorImp & it)
        {
            m_Iterator = it.m_Iterator;
        }

        std::map<std::string, Node *>::iterator m_Iterator;

    };

    class SequenceConstIteratorImp : public IteratorImp
    {

    public:

        virtual Node::eType GetType() const
        {
            return Node::SequenceType;
        }

        virtual void InitBegin(SequenceImp * pSequenceImp)
        {
            m_Iterator = pSequenceImp->m_Sequence.begin();
        }

        virtual void InitEnd(SequenceImp * pSequenceImp)
        {
            m_Iterator = pSequenceImp->m_Sequence.end();
        }

        virtual void InitBegin(MapImp * pMapImp)
        {
        }

        virtual void InitEnd(MapImp * pMapImp)
        {
        }

        void Copy(const SequenceConstIteratorImp & it)
        {
            m_Iterator = it.m_Iterator;
        }

        std::map<size_t, Node *>::const_iterator m_Iterator;

    };

    class MapConstIteratorImp : public IteratorImp
    {

    public:

        virtual Node::eType GetType() const
        {
            return Node::MapType;
        }

        virtual void InitBegin(SequenceImp * pSequenceImp)
        {
        }

        virtual void InitEnd(SequenceImp * pSequenceImp)
        {
        }

        virtual void InitBegin(MapImp * pMapImp)
        {
            m_Iterator = pMapImp->m_Map.begin();
        }

        virtual void InitEnd(MapImp * pMapImp)
        {
            m_Iterator = pMapImp->m_Map.end();
        }

        void Copy(const MapConstIteratorImp & it)
        {
            m_Iterator = it.m_Iterator;
        }

        std::map<std::string, Node *>::const_iterator m_Iterator;

    };


    // Iterator class
    Iterator::Iterator() :
        m_Type(None),
        m_pImp(nullptr)
    {
    }

    Iterator::~Iterator()
    {
        if(m_pImp)
        {
            switch(m_Type)
            {
            case SequenceType:
                delete static_cast<SequenceIteratorImp*>(m_pImp);
                break;
            case MapType:
                delete static_cast<MapIteratorImp*>(m_pImp);
                break;
            default:
                break;
            }

        }
    }

    Iterator::Iterator(const Iterator & it) :
        m_Type(None),
        m_pImp(nullptr)
    {
        *this = it;
    }

    Iterator & Iterator::operator = (const Iterator & it)
    {
        if(m_pImp)
        {
            switch(m_Type)
            {
            case SequenceType:
                delete static_cast<SequenceIteratorImp*>(m_pImp);
                break;
            case MapType:
                delete static_cast<MapIteratorImp*>(m_pImp);
                break;
            default:
                break;
            }
            m_pImp = nullptr;
            m_Type = None;
        }

        IteratorImp * pNewImp = nullptr;

        switch(it.m_Type)
        {
        case SequenceType:
            m_Type = SequenceType;
            pNewImp = new SequenceIteratorImp;
            static_cast<SequenceIteratorImp*>(pNewImp)->m_Iterator = static_cast<SequenceIteratorImp*>(it.m_pImp)->m_Iterator;
            break;
        case MapType:
            m_Type = MapType;
            pNewImp = new MapIteratorImp;
            static_cast<MapIteratorImp*>(pNewImp)->m_Iterator = static_cast<MapIteratorImp*>(it.m_pImp)->m_Iterator;
            break;
        default:
            break;
        }

        m_pImp = pNewImp;
        return *this;
    }

    std::pair<const std::string &, Node &> Iterator::operator *()
    {
        switch(m_Type)
        {
        case SequenceType:
            return { g_EmptyString, *(static_cast<SequenceIteratorImp*>(m_pImp)->m_Iterator->second)};
            break;
        case MapType:
            return {static_cast<MapIteratorImp*>(m_pImp)->m_Iterator->first,
                    *(static_cast<MapIteratorImp*>(m_pImp)->m_Iterator->second)};
            break;
        default:
            break;
        }

        g_NoneNode.Clear();
        return { g_EmptyString, g_NoneNode};
    }

    Iterator & Iterator::operator ++ (int dummy)
    {
        switch(m_Type)
        {
        case SequenceType:
            static_cast<SequenceIteratorImp*>(m_pImp)->m_Iterator++;
            break;
        case MapType:
            static_cast<MapIteratorImp*>(m_pImp)->m_Iterator++;
            break;
        default:
            break;
        }
        return *this;
    }

    Iterator & Iterator::operator -- (int dummy)
    {
        switch(m_Type)
        {
        case SequenceType:
            static_cast<SequenceIteratorImp*>(m_pImp)->m_Iterator--;
            break;
        case MapType:
            static_cast<MapIteratorImp*>(m_pImp)->m_Iterator--;
            break;
        default:
            break;
        }
        return *this;
    }

    bool Iterator::operator == (const Iterator & it)
    {
        if(m_Type != it.m_Type)
        {
            return false;
        }

        switch(m_Type)
        {
        case SequenceType:
            return static_cast<SequenceIteratorImp*>(m_pImp)->m_Iterator == static_cast<SequenceIteratorImp*>(it.m_pImp)->m_Iterator;
            break;
        case MapType:
            return static_cast<MapIteratorImp*>(m_pImp)->m_Iterator == static_cast<MapIteratorImp*>(it.m_pImp)->m_Iterator;
            break;
        default:
            break;
        }

        return false;
    }

    bool Iterator::operator != (const Iterator & it)
    {
        return !(*this == it);
    }


    // Const Iterator class
    ConstIterator::ConstIterator() :
        m_Type(None),
        m_pImp(nullptr)
    {
    }

    ConstIterator::~ConstIterator()
    {
        if(m_pImp)
        {
            switch(m_Type)
            {
            case SequenceType:
                delete static_cast<SequenceConstIteratorImp*>(m_pImp);
                break;
            case MapType:
                delete static_cast<MapConstIteratorImp*>(m_pImp);
                break;
            default:
                break;
            }

        }
    }

    ConstIterator::ConstIterator(const ConstIterator & it) :
        m_Type(None),
        m_pImp(nullptr)
    {
        *this = it;
    }

    ConstIterator & ConstIterator::operator = (const ConstIterator & it)
    {
        if(m_pImp)
        {
            switch(m_Type)
            {
            case SequenceType:
                delete static_cast<SequenceConstIteratorImp*>(m_pImp);
                break;
            case MapType:
                delete static_cast<MapConstIteratorImp*>(m_pImp);
                break;
            default:
                break;
            }
            m_pImp = nullptr;
            m_Type = None;
        }

        IteratorImp * pNewImp = nullptr;

        switch(it.m_Type)
        {
        case SequenceType:
            m_Type = SequenceType;
            pNewImp = new SequenceConstIteratorImp;
            static_cast<SequenceConstIteratorImp*>(pNewImp)->m_Iterator = static_cast<SequenceConstIteratorImp*>(it.m_pImp)->m_Iterator;
            break;
        case MapType:
            m_Type = MapType;
            pNewImp = new MapConstIteratorImp;
            static_cast<MapConstIteratorImp*>(pNewImp)->m_Iterator = static_cast<MapConstIteratorImp*>(it.m_pImp)->m_Iterator;
            break;
        default:
            break;
        }

        m_pImp = pNewImp;
        return *this;
    }

    std::pair<const std::string &, const Node &> ConstIterator::operator *()
    {
        switch(m_Type)
        {
        case SequenceType:
            return { g_EmptyString, *(static_cast<SequenceConstIteratorImp*>(m_pImp)->m_Iterator->second)};
            break;
        case MapType:
            return {static_cast<MapConstIteratorImp*>(m_pImp)->m_Iterator->first,
                    *(static_cast<MapConstIteratorImp*>(m_pImp)->m_Iterator->second)};
            break;
        default:
            break;
        }

        g_NoneNode.Clear();
        return { g_EmptyString, g_NoneNode};
    }

    ConstIterator & ConstIterator::operator ++ (int dummy)
    {
        switch(m_Type)
        {
        case SequenceType:
            static_cast<SequenceConstIteratorImp*>(m_pImp)->m_Iterator++;
            break;
        case MapType:
            static_cast<MapConstIteratorImp*>(m_pImp)->m_Iterator++;
            break;
        default:
            break;
        }
        return *this;
    }

    ConstIterator & ConstIterator::operator -- (int dummy)
    {
        switch(m_Type)
        {
        case SequenceType:
            static_cast<SequenceConstIteratorImp*>(m_pImp)->m_Iterator--;
            break;
        case MapType:
            static_cast<MapConstIteratorImp*>(m_pImp)->m_Iterator--;
            break;
        default:
            break;
        }
        return *this;
    }

    bool ConstIterator::operator == (const ConstIterator & it)
    {
        if(m_Type != it.m_Type)
        {
            return false;
        }

        switch(m_Type)
        {
        case SequenceType:
            return static_cast<SequenceConstIteratorImp*>(m_pImp)->m_Iterator == static_cast<SequenceConstIteratorImp*>(it.m_pImp)->m_Iterator;
            break;
        case MapType:
            return static_cast<MapConstIteratorImp*>(m_pImp)->m_Iterator == static_cast<MapConstIteratorImp*>(it.m_pImp)->m_Iterator;
            break;
        default:
            break;
        }

        return false;
    }

    bool ConstIterator::operator != (const ConstIterator & it)
    {
        return !(*this == it);
    }


    // Node class
    Node::Node() :
        m_pImp(new NodeImp)
    {
    }

    Node::Node(const Node & node) :
        Node()
    {
        *this = node;
    }

    Node::Node(const std::string & value) :
        Node()
    {
        *this = value;
    }

    Node::Node(const char * value) :
        Node()
    {
        *this = value;
    }

    Node::~Node()
    {
        delete static_cast<NodeImp*>(m_pImp);
    }

    Node::eType Node::Type() const
    {
        return NODE_IMP->m_Type;
    }

    bool Node::IsNone() const
    {
        return NODE_IMP->m_Type == Node::None;
    }

    bool Node::IsSequence() const
    {
        return NODE_IMP->m_Type == Node::SequenceType;
    }

    bool Node::IsMap() const
    {
        return NODE_IMP->m_Type == Node::MapType;
    }

    bool Node::IsScalar() const
    {
        return NODE_IMP->m_Type == Node::ScalarType;
    }

    void Node::Clear()
    {
        NODE_IMP->Clear();
    }

    size_t Node::Size() const
    {
        if(TYPE_IMP == nullptr)
        {
            return 0;
        }

        return TYPE_IMP->GetSize();
    }

    Node & Node::Insert(const size_t index)
    {
        NODE_IMP->InitSequence();
        return *TYPE_IMP->Insert(index);
    }

    Node & Node::PushFront()
    {
        NODE_IMP->InitSequence();
        return *TYPE_IMP->PushFront();
    }
    Node & Node::PushBack()
    {
        NODE_IMP->InitSequence();
        return *TYPE_IMP->PushBack();
    }

    Node & Node::operator[](const size_t index)
    {
        NODE_IMP->InitSequence();
        Node * pNode = TYPE_IMP->GetNode(index);
        if(pNode == nullptr)
        {
            g_NoneNode.Clear();
            return g_NoneNode;
        }
        return *pNode;
    }

    Node & Node::operator[](const std::string & key)
    {
        NODE_IMP->InitMap();
        return *TYPE_IMP->GetNode(key);
    }

    void Node::Erase(const size_t index)
    {
        if(TYPE_IMP == nullptr || NODE_IMP->m_Type != Node::SequenceType)
        {
            return;
        }

        return TYPE_IMP->Erase(index);
    }

    void Node::Erase(const std::string & key)
    {
        if(TYPE_IMP == nullptr || NODE_IMP->m_Type != Node::MapType)
        {
            return;
        }

        return TYPE_IMP->Erase(key);
    }

    Node & Node::operator = (const Node & node)
    {
        NODE_IMP->Clear();
        CopyNode(node, *this);
        return *this;
    }

    Node & Node::operator = (const std::string & value)
    {
        NODE_IMP->InitScalar();
        TYPE_IMP->SetData(value);
        return *this;
    }

    Node & Node::operator = (const char * value)
    {
        NODE_IMP->InitScalar();
        TYPE_IMP->SetData(value ? std::string(value) : "");
        return *this;
    }

    Iterator Node::Begin()
    {
        Iterator it;

        if(TYPE_IMP != nullptr)
        {
            IteratorImp * pItImp = nullptr;

            switch(NODE_IMP->m_Type)
            {
            case Node::SequenceType:
                it.m_Type = Iterator::SequenceType;
                pItImp = new SequenceIteratorImp;
                pItImp->InitBegin(static_cast<SequenceImp*>(TYPE_IMP));
                break;
            case Node::MapType:
                it.m_Type = Iterator::MapType;
                pItImp = new MapIteratorImp;
                pItImp->InitBegin(static_cast<MapImp*>(TYPE_IMP));
                break;
            default:
                break;
            }

            it.m_pImp = pItImp;
        }

        return it;
    }

    ConstIterator Node::Begin() const
    {
        ConstIterator it;

        if(TYPE_IMP != nullptr)
        {
            IteratorImp * pItImp = nullptr;

            switch(NODE_IMP->m_Type)
            {
            case Node::SequenceType:
                it.m_Type = ConstIterator::SequenceType;
                pItImp = new SequenceConstIteratorImp;
                pItImp->InitBegin(static_cast<SequenceImp*>(TYPE_IMP));
                break;
            case Node::MapType:
                it.m_Type = ConstIterator::MapType;
                pItImp = new MapConstIteratorImp;
                pItImp->InitBegin(static_cast<MapImp*>(TYPE_IMP));
                break;
            default:
                break;
            }

            it.m_pImp = pItImp;
        }

        return it;
    }

    Iterator Node::End()
    {
       Iterator it;

        if(TYPE_IMP != nullptr)
        {
            IteratorImp * pItImp = nullptr;

            switch(NODE_IMP->m_Type)
            {
            case Node::SequenceType:
                it.m_Type = Iterator::SequenceType;
                pItImp = new SequenceIteratorImp;
                pItImp->InitEnd(static_cast<SequenceImp*>(TYPE_IMP));
                break;
            case Node::MapType:
                it.m_Type = Iterator::MapType;
                pItImp = new MapIteratorImp;
                pItImp->InitEnd(static_cast<MapImp*>(TYPE_IMP));
                break;
            default:
                break;
            }

            it.m_pImp = pItImp;
        }

        return it;
    }

    ConstIterator Node::End() const
    {
       ConstIterator it;

        if(TYPE_IMP != nullptr)
        {
            IteratorImp * pItImp = nullptr;

            switch(NODE_IMP->m_Type)
            {
            case Node::SequenceType:
                it.m_Type = ConstIterator::SequenceType;
                pItImp = new SequenceConstIteratorImp;
                pItImp->InitEnd(static_cast<SequenceImp*>(TYPE_IMP));
                break;
            case Node::MapType:
                it.m_Type = ConstIterator::MapType;
                pItImp = new MapConstIteratorImp;
                pItImp->InitEnd(static_cast<MapImp*>(TYPE_IMP));
                break;
            default:
                break;
            }

            it.m_pImp = pItImp;
        }

        return it;
    }

    const std::string & Node::AsString() const
    {
        if(TYPE_IMP == nullptr)
        {
            return g_EmptyString;
        }

        return TYPE_IMP->GetData();
    }



    // Reader implementations
    /**
    * @breif Line information structure.
    *
    */
    class ReaderLine
    {

    public:

        /**
        * @breif Constructor.
        *
        */
        ReaderLine(const std::string & data = "",
                   const size_t no = 0,
                   const size_t offset = 0,
                   const Node::eType type = Node::None,
                   const unsigned char flags = 0) :
            Data(data),
            No(no),
            Offset(offset),
            Type(type),
            Flags(flags),
            NextLine(nullptr)
        {
        }

        enum eFlag
        {
            LiteralScalarFlag,      ///< Literal scalar type, defined as "|".
            FoldedScalarFlag,       ///< Folded scalar type, defined as "<".
            ScalarNewlineFlag       ///< Scalar ends with a newline.
        };

        /**
        * @breif Set flag.
        *
        */
        void SetFlag(const eFlag flag)
        {
            Flags |= FlagMask[static_cast<size_t>(flag)];
        }

        /**
        * @breif Set flags by mask value.
        *
        */
        void SetFlags(const unsigned char flags)
        {
            Flags |= flags;
        }

        /**
        * @breif Unset flag.
        *
        */
        void UnsetFlag(const eFlag flag)
        {
            Flags &= ~FlagMask[static_cast<size_t>(flag)];
        }

        /**
        * @breif Unset flags by mask value.
        *
        */
        void UnsetFlags(const unsigned char flags)
        {
            Flags &= ~flags;
        }

        /**
        * @breif Get flag value.
        *
        */
        bool GetFlag(const eFlag flag) const
        {
            return Flags & FlagMask[static_cast<size_t>(flag)];
        }

        /**
        * @breif Copy and replace scalar flags from another ReaderLine.
        *
        */
        void CopyScalarFlags(ReaderLine * from)
        {
            if (from == nullptr)
            {
                return;
            }

            unsigned char newFlags = from->Flags & (FlagMask[0] | FlagMask[1] | FlagMask[2]);
            Flags |= newFlags;
        }

        static const unsigned char FlagMask[3];

        std::string     Data;       ///< Data of line.
        size_t          No;         ///< Line number.
        size_t          Offset;     ///< Offset to first character in data.
        Node::eType     Type;       ///< Type of line.
        unsigned char   Flags;      ///< Flags of line.
        ReaderLine *    NextLine;   ///< Pointer to next line.



    };

    const unsigned char ReaderLine::FlagMask[3] = { 0x01, 0x02, 0x04 };


    /**
    * @breif Implementation class of Yaml parsing.
    *        Parsing incoming stream and outputs a root node.
    *
    */
    class ParseImp
    {

    public:

        /**
        * @breif Default constructor.
        *
        */
        ParseImp()
        {
        }

        /**
        * @breif Destructor.
        *
        */
        ~ParseImp()
        {
            ClearLines();
        }

        /**
        * @breif Run full parsing procedure.
        *
        */
        void Parse(Node & root, std::iostream & stream)
        {
            try
            {
                root.Clear();
                ReadLines(stream);
                PostProcessLines();
                //Print();
                ParseRoot(root);
            }
            catch(Exception e)
            {
                root.Clear();
                throw;
            }
        }

    private:

        /**
        * @breif Copy constructor.
        *
        */
        ParseImp(const ParseImp & copy)
        {

        }

        /**
        * @breif Read all lines.
        *        Ignoring:
        *           - Empty lines.
        *           - Comments.
        *           - Document start/end.
        *
        */
        void ReadLines(std::iostream & stream)
        {
            std::string     line = "";
            size_t          lineNo = 0;
            bool            documentStartFound = false;
            bool            foundFirstNotEmpty = false;
            std::streampos  streamPos = 0;

            // Read all lines, as long as the stream is ok.
            while (!stream.eof() && !stream.fail())
            {
                // Read line
                streamPos = stream.tellg();
                std::getline(stream, line);
                lineNo++;

                // Remove comment
                const size_t commentPos = FindNotCited(line, '#');
                if(commentPos != std::string::npos)
                {
                    line.resize(commentPos);
                }

                // Start of document.
                if (documentStartFound == false && line == "---")
                {
                    // Erase all lines before this line.
                    ClearLines();
                    documentStartFound = true;
                    continue;
                }

                // End of document.
                if (line == "...")
                {
                    break;
                }
                else if(line == "---")
                {
                    stream.seekg(streamPos);
                    break;
                }

                // Remove trailing return.
                if (line.size())
                {
                    if (line[line.size() - 1] == '\r')
                    {
                        line.resize(line.size() - 1);
                    }
                }

                // Validate characters.
                for (size_t i = 0; i < line.size(); i++)
                {
                    if (line[i] != '\t' && (line[i] < 32 || line[i] > 125))
                    {
                        throw ParsingException(ExceptionMessage(g_ErrorInvalidCharacter, lineNo, i + 1));
                    }
                }

                // Validate tabs
                const size_t firstTabPos    = line.find_first_of('\t');
                size_t       startOffset    = line.find_first_not_of(" \t");

                // Make sure no tabs are in the very front.
                if (startOffset != std::string::npos)
                {
                    if(firstTabPos < startOffset)
                    {
                        throw ParsingException(ExceptionMessage(g_ErrorTabInOffset, lineNo, firstTabPos));
                    }

                    // Remove front spaces.
                    line = line.substr(startOffset);
                }
                else
                {
                    startOffset = 0;
                    line = "";
                }

                // Add line.
                if(foundFirstNotEmpty == false)
                {
                    if(line.size())
                    {
                        foundFirstNotEmpty = true;
                    }
                    else
                    {
                        continue;
                    }
                }

                ReaderLine * pLine = new ReaderLine(line, lineNo, startOffset);
                m_Lines.push_back(pLine);
            }
        }

        /**
        * @breif Run post-processing on all lines.
        *        Basically split lines into multiple lines if needed, to follow the parsing algorithm.
        *
        */
        void PostProcessLines()
        {
            for (auto it = m_Lines.begin(); it != m_Lines.end();)
            {
                // Sequence.
                if (PostProcessSequenceLine(it) == true)
                {
                    continue;
                }

                // Mapping.
                if (PostProcessMappingLine(it) == true)
                {
                    continue;
                }

                // Scalar.
                PostProcessScalarLine(it);
            }

            // Set next line of all lines.
            if (m_Lines.size())
            {
                if (m_Lines.back()->Type != Node::ScalarType)
                {
                    throw ParsingException(ExceptionMessage(g_ErrorUnexpectedDocumentEnd, *m_Lines.back()));
                }

                if (m_Lines.size() > 1)
                {
                    auto prevEnd = m_Lines.end();
                    --prevEnd;

                    for (auto it = m_Lines.begin(); it != prevEnd; it++)
                    {
                        auto nextIt = it;
                        ++nextIt;

                        (*it)->NextLine = *nextIt;
                    }
                }
            }
        }

        /**
        * @breif Run post-processing and check for sequence.
        *        Split line into two lines if sequence token is not on it's own line.
        *
        * @return true if line is sequence, else false.
        *
        */
        bool PostProcessSequenceLine(std::list<ReaderLine *>::iterator & it)
        {
            ReaderLine * pLine = *it;

            // Sequence split
            if (IsSequenceStart(pLine->Data) == false)
            {
                return false;
            }

            pLine->Type = Node::SequenceType;

            ClearTrailingEmptyLines(++it);

            const size_t valueStart = pLine->Data.find_first_not_of(" \t", 1);
            if (valueStart == std::string::npos)
            {
                return true;
            }

            // Create new line and insert
            std::string newLine = pLine->Data.substr(valueStart);
            it = m_Lines.insert(it, new ReaderLine(newLine, pLine->No, pLine->Offset + valueStart));
            pLine->Data = "";

            return false;
        }

        /**
        * @breif Run post-processing and check for mapping.
        *        Split line into two lines if mapping value is not on it's own line.
        *
        * @return true if line is mapping, else move on to scalar parsing.
        *
        */
        bool PostProcessMappingLine(std::list<ReaderLine *>::iterator & it)
        {
            ReaderLine * pLine = *it;

            // Find map key.
            size_t preKeyQuotes = 0;
            size_t tokenPos = FindNotCited(pLine->Data, ':', preKeyQuotes);
            if (tokenPos == std::string::npos)
            {
                return false;
            }
            if(preKeyQuotes > 1)
            {
                throw ParsingException(ExceptionMessage(g_ErrorKeyIncorrect, *pLine));
            }

            pLine->Type = Node::MapType;

            // Get key
            std::string key = pLine->Data.substr(0, tokenPos);
            const size_t keyEnd = key.find_last_not_of(" \t");
            if (keyEnd == std::string::npos)
            {
                throw ParsingException(ExceptionMessage(g_ErrorKeyMissing, *pLine));
            }
            key.resize(keyEnd + 1);

            // Handle cited key.
            if(preKeyQuotes == 1)
            {
                if(key.front() != '"' || key.back() != '"')
                {
                    throw ParsingException(ExceptionMessage(g_ErrorKeyIncorrect, *pLine));
                }

                key = key.substr(1, key.size() - 2);
            }
            RemoveAllEscapeTokens(key);

            // Get value
            std::string value = "";
            size_t valueStart = std::string::npos;
            if (tokenPos + 1 != pLine->Data.size())
            {
                valueStart = pLine->Data.find_first_not_of(" \t", tokenPos + 1);
                if (valueStart != std::string::npos)
                {
                    value = pLine->Data.substr(valueStart);
                }
            }

            // Make sure the value is not a sequence start.
            if (IsSequenceStart(value) == true)
            {
                throw ParsingException(ExceptionMessage(g_ErrorBlockSequenceNotAllowed, *pLine, valueStart));
            }

            pLine->Data = key;


            // Remove all empty lines after map key.
            ClearTrailingEmptyLines(++it);

            // Add new empty line?
            size_t newLineOffset = valueStart;
            if(newLineOffset == std::string::npos)
            {
                if(it != m_Lines.end() && (*it)->Offset > pLine->Offset)
                {
                    return true;
                }

                newLineOffset = tokenPos + 2;
            }
            else
            {
                newLineOffset += pLine->Offset;
            }

            // Add new line with value.
            unsigned char dummyBlockFlags = 0;
            if(IsBlockScalar(value, pLine->No, dummyBlockFlags) == true)
            {
                newLineOffset = pLine->Offset;
            }
            ReaderLine * pNewLine = new ReaderLine(value, pLine->No, newLineOffset, Node::ScalarType);
            it = m_Lines.insert(it, pNewLine);

            // Return false in order to handle next line(scalar value).
            return false;
        }

        /**
        * @breif Run post-processing and check for scalar.
        *        Checking for multi-line scalars.
        *
        * @return true if scalar search should continue, else false.
        *
        */
        void PostProcessScalarLine(std::list<ReaderLine *>::iterator & it)
        {
            ReaderLine * pLine = *it;
            pLine->Type = Node::ScalarType;

            size_t parentOffset = pLine->Offset;
            if(pLine != m_Lines.front())
            {
                 std::list<ReaderLine *>::iterator lastIt = it;
                --lastIt;
                parentOffset = (*lastIt)->Offset;
            }

           std::list<ReaderLine *>::iterator lastNotEmpty = it++;

            // Find last empty lines
            while(it != m_Lines.end())
            {
                pLine = *it;
                pLine->Type = Node::ScalarType;
                if(pLine->Data.size())
                {
                    if(pLine->Offset <= parentOffset)
                    {
                        break;
                    }
                    else
                    {
                        lastNotEmpty = it;
                    }
                }
                ++it;
            }

            ClearTrailingEmptyLines(++lastNotEmpty);
        }

        /**
        * @breif Process root node and start of document.
        *
        */
        void ParseRoot(Node & root)
        {
            // Get first line and start type.
            auto it = m_Lines.begin();
            if(it == m_Lines.end())
            {
                return;
            }
            Node::eType type = (*it)->Type;
            ReaderLine * pLine = *it;

            // Handle next line.
            switch(type)
            {
            case Node::SequenceType:
                ParseSequence(root, it);
                break;
            case Node::MapType:
                ParseMap(root, it);
                break;
            case Node::ScalarType:
                ParseScalar(root, it);
                break;
            default:
                break;
            }

            if(it != m_Lines.end())
            {
                throw InternalException(ExceptionMessage(g_ErrorUnexpectedDocumentEnd, *pLine));
            }

        }

        /**
        * @breif Process sequence node.
        *
        */
        void ParseSequence(Node & node, std::list<ReaderLine *>::iterator & it)
        {
            ReaderLine * pNextLine = nullptr;
            while(it != m_Lines.end())
            {
                ReaderLine * pLine = *it;
                Node & childNode = node.PushBack();

                // Move to next line, error check.
                ++it;
                if(it == m_Lines.end())
                {
                    throw InternalException(ExceptionMessage(g_ErrorUnexpectedDocumentEnd, *pLine));
                }

                // Handle value of map
                Node::eType valueType = (*it)->Type;
                switch(valueType)
                {
                case Node::SequenceType:
                    ParseSequence(childNode, it);
                    break;
                case Node::MapType:
                    ParseMap(childNode, it);
                    break;
                case Node::ScalarType:
                    ParseScalar(childNode, it);
                    break;
                default:
                    break;
                }

                // Check next line. if sequence and correct level, go on, else exit.
                // If same level but but of type map = error.
                if(it == m_Lines.end() || ((pNextLine = *it)->Offset < pLine->Offset))
                {
                    break;
                }
                if(pNextLine->Offset > pLine->Offset)
                {
                    throw ParsingException(ExceptionMessage(g_ErrorIncorrectOffset, *pNextLine));
                }
                if(pNextLine->Type != Node::SequenceType)
                {
                    throw InternalException(ExceptionMessage(g_ErrorDiffEntryNotAllowed, *pNextLine));
                }

            }
        }

        /**
        * @breif Process map node.
        *
        */
        void ParseMap(Node & node, std::list<ReaderLine *>::iterator & it)
        {
            ReaderLine * pNextLine = nullptr;
            while(it != m_Lines.end())
            {
                ReaderLine * pLine = *it;
                Node & childNode = node[pLine->Data];

                // Move to next line, error check.
                ++it;
                if(it == m_Lines.end())
                {
                    throw InternalException(ExceptionMessage(g_ErrorUnexpectedDocumentEnd, *pLine));
                }

                // Handle value of map
                Node::eType valueType = (*it)->Type;
                switch(valueType)
                {
                case Node::SequenceType:
                    ParseSequence(childNode, it);
                    break;
                case Node::MapType:
                    ParseMap(childNode, it);
                    break;
                case Node::ScalarType:
                    ParseScalar(childNode, it);
                    break;
                default:
                    break;
                }

                // Check next line. if map and correct level, go on, else exit.
                // if same level but but of type map = error.
                if(it == m_Lines.end() || ((pNextLine = *it)->Offset < pLine->Offset))
                {
                    break;
                }
                if(pNextLine->Offset > pLine->Offset)
                {
                    throw ParsingException(ExceptionMessage(g_ErrorIncorrectOffset, *pNextLine));
                }
                if(pNextLine->Type != pLine->Type)
                {
                    throw InternalException(ExceptionMessage(g_ErrorDiffEntryNotAllowed, *pNextLine));
                }

            }
        }

        /**
        * @breif Process scalar node.
        *
        */
        void ParseScalar(Node & node, std::list<ReaderLine *>::iterator & it)
        {
            std::string data = "";
            ReaderLine * pFirstLine = *it;
            ReaderLine * pLine = *it;

            // Check if current line is a block scalar.
            unsigned char blockFlags = 0;
            bool isBlockScalar = IsBlockScalar(pLine->Data, pLine->No, blockFlags);
            const bool newLineFlag = static_cast<bool>(blockFlags & ReaderLine::FlagMask[static_cast<size_t>(ReaderLine::ScalarNewlineFlag)]);
            const bool foldedFlag = static_cast<bool>(blockFlags & ReaderLine::FlagMask[static_cast<size_t>(ReaderLine::FoldedScalarFlag)]);
            const bool literalFlag = static_cast<bool>(blockFlags & ReaderLine::FlagMask[static_cast<size_t>(ReaderLine::LiteralScalarFlag)]);
            size_t parentOffset = 0;

            // Find parent offset
            if(it != m_Lines.begin())
            {
                std::list<ReaderLine *>::iterator parentIt = it;
                --parentIt;
                parentOffset = (*parentIt)->Offset;
            }

            // Move to next iterator/line if current line is a block scalar.
            if(isBlockScalar)
            {
                ++it;
                if(it == m_Lines.end() || (pLine = *it)->Type != Node::ScalarType)
                {
                    return;
                }
            }

            // Not a block scalar, cut end spaces/tabs
            if(isBlockScalar == false)
            {
                while(1)
                {
                    pLine = *it;

                    if(parentOffset != 0 && pLine->Offset <= parentOffset)
                    {
                        throw ParsingException(ExceptionMessage(g_ErrorIncorrectOffset, *pLine));
                    }

                    const size_t endOffset = pLine->Data.find_last_not_of(" \t");
                    if(endOffset == std::string::npos)
                    {
                        data += "\n";
                    }
                    else
                    {
                        data += pLine->Data.substr(0, endOffset + 1);
                    }

                    // Move to next line
                    ++it;
                    if(it == m_Lines.end() || (*it)->Type != Node::ScalarType)
                    {
                        break;
                    }

                    data += " ";
                }

                if(ValidateQuote(data) == false)
                {
                    throw ParsingException(ExceptionMessage(g_ErrorInvalidQuote, *pFirstLine));
                }
            }
            // Block scalar
            else
            {
                pLine = *it;
                size_t blockOffset = pLine->Offset;
                if(blockOffset <= parentOffset)
                {
                    throw ParsingException(ExceptionMessage(g_ErrorIncorrectOffset, *pLine));
                }

                bool addedSpace = false;
                while(it != m_Lines.end() && (*it)->Type == Node::ScalarType)
                {
                    pLine = *it;

                    const size_t endOffset = pLine->Data.find_last_not_of(" \t");
                    if(endOffset != std::string::npos && pLine->Offset < blockOffset)
                    {
                        throw ParsingException(ExceptionMessage(g_ErrorIncorrectOffset, *pLine));
                    }

                    if(endOffset == std::string::npos)
                    {
                        if(addedSpace)
                        {
                            data[data.size() - 1] = '\n';
                            addedSpace = false;
                        }
                        else
                        {
                            data += "\n";
                        }

                        ++it;
                        continue;
                    }
                    else
                    {
                        if(blockOffset != pLine->Offset && foldedFlag)
                        {
                            if(addedSpace)
                            {
                                data[data.size() - 1] = '\n';
                                addedSpace = false;
                            }
                            else
                            {
                                data += "\n";
                            }
                        }
                        data += std::string(pLine->Offset - blockOffset, ' ');
                        data += pLine->Data;
                    }

                    // Move to next line
                    ++it;
                    if(it == m_Lines.end() || (*it)->Type != Node::ScalarType)
                    {
                        if(newLineFlag)
                        {
                            data += "\n";
                        }
                        break;
                    }

                    if(foldedFlag)
                    {
                        data += " ";
                        addedSpace = true;
                    }
                    else if(literalFlag && endOffset != std::string::npos)
                    {
                        data += "\n";
                    }
                }
            }

            if(data.size() && (data[0] == '"' || data[0] == '\''))
            {
                data = data.substr(1, data.size() - 2 );
            }

            node = data;
        }

        /**
        * @breif Debug printing.
        *
        */
        void Print()
        {
            for (auto it = m_Lines.begin(); it != m_Lines.end(); it++)
            {

                ReaderLine * pLine = *it;

                // Print type
                if (pLine->Type == Node::SequenceType)
                {
                    std::cout << "seq ";
                }
                else if (pLine->Type == Node::MapType)
                {
                    std::cout << "map ";
                }
                else if (pLine->Type == Node::ScalarType)
                {
                    std::cout << "sca ";
                }
                else
                {
                    std::cout << "    ";
                }

                // Print flags
                if (pLine->GetFlag(ReaderLine::FoldedScalarFlag))
                {
                    std::cout << "f";
                }
                else
                {
                    std::cout << "-";
                }
                if (pLine->GetFlag(ReaderLine::LiteralScalarFlag))
                {
                    std::cout << "l";
                }
                else
                {
                    std::cout << "-";
                }
                if (pLine->GetFlag(ReaderLine::ScalarNewlineFlag))
                {
                    std::cout << "n";
                }
                else
                {
                    std::cout << "-";
                }
                if (pLine->NextLine == nullptr)
                {
                    std::cout << "e";
                }
                else
                {
                    std::cout << "-";
                }


                std::cout << "| ";
                std::cout << pLine->No << " ";
                std::cout << std::string(pLine->Offset, ' ');

                if (pLine->Type == Node::ScalarType)
                {
                    std::string scalarValue = pLine->Data;
                    for (size_t i = 0; (i = scalarValue.find("\n", i)) != std::string::npos;)
                    {
                        scalarValue.replace(i, 1, "\\n");
                        i += 2;
                    }
                    std::cout << scalarValue << std::endl;
                }
                else if (pLine->Type == Node::MapType)
                {
                    std::cout << pLine->Data + ":" << std::endl;
                }
                else if (pLine->Type == Node::SequenceType)
                {
                    std::cout << "-" << std::endl;
                }
                else
                {
                    std::cout << "> UNKOWN TYPE <" << std::endl;
                }
            }
        }

        /**
        * @breif Clear all read lines.
        *
        */
        void ClearLines()
        {
            for (auto it = m_Lines.begin(); it != m_Lines.end(); it++)
            {
                delete *it;
            }
            m_Lines.clear();
        }

        void ClearTrailingEmptyLines(std::list<ReaderLine *>::iterator & it)
        {
            while(it != m_Lines.end())
            {
                ReaderLine * pLine = *it;
                if(pLine->Data.size() == 0)
                {
                    delete *it;
                    it = m_Lines.erase(it);
                }
                else
                {
                    return;
                }

            }
        }

        static bool IsSequenceStart(const std::string & data)
        {
            if (data.size() == 0 || data[0] != '-')
            {
                return false;
            }

            if (data.size() >= 2 && data[1] != ' ')
            {
                return false;
            }

            return true;
        }

        static bool IsBlockScalar(const std::string & data, const size_t line, unsigned char & flags)
        {
            flags = 0;
            if(data.size() == 0)
            {
                return false;
            }

            if(data[0] == '|')
            {
                if(data.size() >= 2)
                {
                    if(data[1] != '-' && data[1] != ' ' && data[1] != '\t')
                    {
                        throw ParsingException(ExceptionMessage(g_ErrorInvalidBlockScalar, line, data));
                    }
                }
                else
                {
                    flags |= ReaderLine::FlagMask[static_cast<size_t>(ReaderLine::ScalarNewlineFlag)];
                }
                flags |= ReaderLine::FlagMask[static_cast<size_t>(ReaderLine::LiteralScalarFlag)];
                return true;
            }

            if(data[0] == '>')
            {
                if(data.size() >= 2)
                {
                    if(data[1] != '-' && data[1] != ' ' && data[1] != '\t')
                    {
                        throw ParsingException(ExceptionMessage(g_ErrorInvalidBlockScalar, line, data));
                    }
                }
                else
                {
                    flags |= ReaderLine::FlagMask[static_cast<size_t>(ReaderLine::ScalarNewlineFlag)];
                }
                flags |= ReaderLine::FlagMask[static_cast<size_t>(ReaderLine::FoldedScalarFlag)];
                return true;
            }

            return false;
        }

        std::list<ReaderLine *> m_Lines;    ///< List of lines.

    };

    // Parsing functions
    void Parse(Node & root, const char * filename)
    {
        std::ifstream f(filename, std::ifstream::binary);
        if (f.is_open() == false)
        {
            throw OperationException(g_ErrorCannotOpenFile);
        }

        f.seekg(0, f.end);
        size_t fileSize = static_cast<size_t>(f.tellg());
        f.seekg(0, f.beg);

        std::unique_ptr<char[]> data(new char[fileSize]);
        f.read(data.get(), fileSize);
        f.close();

        Parse(root, data.get(), fileSize);
    }

    void Parse(Node & root, std::iostream & stream)
    {
        ParseImp * pImp = nullptr;

        try
        {
            pImp = new ParseImp;
            pImp->Parse(root, stream);
            delete pImp;
        }
        catch (const Exception e)
        {
            delete pImp;
            throw;
        }
    }

    void Parse(Node & root, const std::string & string)
    {
        std::stringstream ss(string);
        Parse(root, ss);
    }

    void Parse(Node & root, const char * buffer, const size_t size)
    {
        std::stringstream ss(std::string(buffer, size));
        Parse(root, ss);
    }


    // Serialize configuration structure.
    SerializeConfig::SerializeConfig(const size_t spaceIndentation,
                                     const size_t scalarMaxLength,
                                     const bool sequenceMapNewline,
                                     const bool mapScalarNewline) :
        SpaceIndentation(spaceIndentation),
        ScalarMaxLength(scalarMaxLength),
        SequenceMapNewline(sequenceMapNewline),
        MapScalarNewline(mapScalarNewline)
    {
    }


    // Serialization functions
    void Serialize(const Node & root, const char * filename, const SerializeConfig & config)
    {
        std::stringstream stream;
        Serialize(root, stream, config);

        std::ofstream f(filename);
        if (f.is_open() == false)
        {
            throw OperationException(g_ErrorCannotOpenFile);
        }

        f.write(stream.str().c_str(), stream.str().size());
        f.close();
    }

    size_t LineFolding(const std::string & input, std::vector<std::string> & folded, const size_t maxLength)
    {
        folded.clear();
        if(input.size() == 0)
        {
            return 0;
        }

        size_t currentPos = 0;
        size_t lastPos = 0;
        size_t spacePos = std::string::npos;
        while(currentPos < input.size())
        {
            currentPos = lastPos + maxLength;

            if(currentPos < input.size())
            {
               spacePos = input.find_first_of(' ', currentPos);
            }

            if(spacePos == std::string::npos || currentPos >= input.size())
            {
                const std::string endLine = input.substr(lastPos);
                if(endLine.size())
                {
                    folded.push_back(endLine);
                }

                return folded.size();
            }

            folded.push_back(input.substr(lastPos, spacePos - lastPos));

            lastPos = spacePos + 1;
        }

        return folded.size();
    }

    static void SerializeLoop(const Node & node, std::iostream & stream, bool useLevel, const size_t level, const SerializeConfig & config)
    {
        const size_t indention = config.SpaceIndentation;

        switch(node.Type())
        {
            case Node::SequenceType:
            {
                for(auto it = node.Begin(); it != node.End(); it++)
                {
                    const Node & value = (*it).second;
                    if(value.IsNone())
                    {
                        continue;
                    }
                    stream << std::string(level, ' ') << "- ";
                    useLevel = false;
                    if(value.IsSequence() || (value.IsMap() && config.SequenceMapNewline == true))
                    {
                        useLevel = true;
                        stream << "\n";
                    }

                    SerializeLoop(value, stream, useLevel, level + 2, config);
                }

            }
            break;
            case Node::MapType:
            {
                size_t count = 0;
                for(auto it = node.Begin(); it != node.End(); it++)
                {
                    const Node & value = (*it).second;
                    if(value.IsNone())
                    {
                        continue;
                    }

                    if(useLevel || count > 0)
                    {
                       stream << std::string(level, ' ');
                    }

                    std::string key = (*it).first;
                    AddEscapeTokens(key, "\\\"");
                    if(ShouldBeCited(key))
                    {
                        stream << "\"" << key << "\"" << ": ";
                    }
                    else
                    {
                        stream << key << ": ";
                    }


                    useLevel = false;
                    if(value.IsScalar() == false || (value.IsScalar() && config.MapScalarNewline))
                    {
                        useLevel = true;
                        stream << "\n";
                    }

                    SerializeLoop(value, stream, useLevel, level + indention, config);

                    useLevel = true;
                    count++;
                }

            }
            break;
            case Node::ScalarType:
            {
                const std::string value = node.As<std::string>();

                // Empty scalar
                if(value.size() == 0)
                {
                    stream << "\n";
                    break;
                }

                // Get lines of scalar.
                std::string line = "";
                std::vector<std::string> lines;
                std::istringstream iss(value);
                while (iss.eof() == false)
                {
                    std::getline(iss, line);
                    lines.push_back(line);
                }

                // Block scalar
                const std::string & lastLine = lines.back();
                const bool endNewline = lastLine.size() == 0;
                if(endNewline)
                {
                    lines.pop_back();
                }

                // Literal
                if(lines.size() > 1)
                {
                    stream << "|";
                }
                // Folded/plain
                else
                {
                    const std::string frontLine = lines.front();
                    if(config.ScalarMaxLength == 0 || lines.front().size() <= config.ScalarMaxLength ||
                       LineFolding(frontLine, lines, config.ScalarMaxLength) == 1)
                    {
                        if(useLevel)
                        {
                             stream << std::string(level, ' ');
                        }

                        if(ShouldBeCited(value))
                        {
                            stream << "\"" << value << "\"\n";
                            break;
                        }
                        stream << value << "\n";
                        break;
                    }
                    else
                    {
                        stream << ">";
                    }
                }

                if(endNewline == false)
                {
                     stream << "-";
                }
                stream << "\n";


                for(auto it = lines.begin(); it != lines.end(); it++)
                {
                    stream << std::string(level, ' ') << (*it) << "\n";
                }
            }
            break;

        default:
            break;
        }
    }

    void Serialize(const Node & root, std::iostream & stream, const SerializeConfig & config)
    {
        if(config.SpaceIndentation < 2)
        {
            throw OperationException(g_ErrorIndentation);
        }

        SerializeLoop(root, stream, false, 0, config);
    }

    void Serialize(const Node & root, std::string & string, const SerializeConfig & config)
    {
        std::stringstream stream;
        Serialize(root, stream, config);
        string = stream.str();
    }



    // Static function implementations
    std::string ExceptionMessage(const std::string & message, ReaderLine & line)
    {
        return message + std::string(" Line ") + std::to_string(line.No) + std::string(": ") + line.Data;
    }

    std::string ExceptionMessage(const std::string & message, ReaderLine & line, const size_t errorPos)
    {
        return message + std::string(" Line ") + std::to_string(line.No) + std::string(" column ") + std::to_string(errorPos + 1) + std::string(": ") + line.Data;
    }

    std::string ExceptionMessage(const std::string & message, const size_t errorLine, const size_t errorPos)
    {
        return message + std::string(" Line ") + std::to_string(errorLine) + std::string(" column ") + std::to_string(errorPos);
    }

    std::string ExceptionMessage(const std::string & message, const size_t errorLine, const std::string & data)
    {
        return message + std::string(" Line ") + std::to_string(errorLine) + std::string(": ") + data;
    }

    bool FindQuote(const std::string & input, size_t & start, size_t & end, size_t searchPos)
    {
        start = end = std::string::npos;
        size_t qPos = searchPos;
        bool foundStart = false;

        while(qPos != std::string::npos)
        {
            // Find first quote.
            qPos = input.find_first_of("\"'", qPos);
            if(qPos == std::string::npos)
            {
                return false;
            }

            const char token = input[qPos];
            if(token == '"' && (qPos == 0 || input[qPos-1] != '\\'))
            {
                // Found start quote.
                if(foundStart == false)
                {
                    start = qPos;
                    foundStart = true;
                }
                // Found end quote
                else
                {
                    end = qPos;
                    return true;
                }
            }

            // Check if it's possible for another loop.
            if(qPos + 1 == input.size())
            {
                return false;
            }
            qPos++;
        }

        return false;
    }

    size_t FindNotCited(const std::string & input, char token, size_t & preQuoteCount)
    {
        preQuoteCount = 0;
        size_t tokenPos = input.find_first_of(token);
        if(tokenPos == std::string::npos)
        {
            return std::string::npos;
        }

        // Find all quotes
        std::vector<std::pair<size_t, size_t>> quotes;

        size_t quoteStart = 0;
        size_t quoteEnd = 0;
        while(FindQuote(input, quoteStart, quoteEnd, quoteEnd))
        {
            quotes.push_back({quoteStart, quoteEnd});

            if(quoteEnd + 1 == input.size())
            {
                break;
            }
            quoteEnd++;
        }

        if(quotes.size() == 0)
        {
            return tokenPos;
        }

        size_t currentQuoteIndex = 0;
        std::pair<size_t, size_t> currentQuote = {0, 0};

        while(currentQuoteIndex < quotes.size())
        {
            currentQuote = quotes[currentQuoteIndex];

            if(tokenPos < currentQuote.first)
            {
                return tokenPos;
            }
            preQuoteCount++;
            if(tokenPos <= currentQuote.second)
            {
                // Find next token
                if(tokenPos + 1 == input.size())
                {
                    return std::string::npos;
                }
                tokenPos = input.find_first_of(token, tokenPos + 1);
                if(tokenPos == std::string::npos)
                {
                    return std::string::npos;
                }
            }

            currentQuoteIndex++;
        }

        return tokenPos;
    }

    size_t FindNotCited(const std::string & input, char token)
    {
        size_t dummy = 0;
        return FindNotCited(input, token, dummy);
    }

    bool ValidateQuote(const std::string & input)
    {
        if(input.size() == 0)
        {
            return true;
        }

        char token = 0;
        size_t searchPos = 0;
        if(input[0] == '\"' || input[0] == '\'')
        {
            if(input.size() == 1)
            {
                return false;
            }
            token = input[0];
            searchPos = 1;
        }

        while(searchPos != std::string::npos && searchPos < input.size() - 1)
        {
            searchPos = input.find_first_of("\"'", searchPos + 1);
            if(searchPos == std::string::npos)
            {
                break;
            }

            const char foundToken = input[searchPos];

            if(input[searchPos] == '\"' || input[searchPos] == '\'')
            {
                if(token == 0 && input[searchPos-1] != '\\')
                {
                    return false;
                }
                //if(foundToken == token)
                //{

                    /*if(foundToken == token && searchPos == input.size() - 1 && input[searchPos-1] != '\\')
                    {
                        return true;
                        if(searchPos == input.size() - 1)
                        {
                            return true;
                        }
                        return false;
                    }
                    else */
                    if(foundToken == token && input[searchPos-1] != '\\')
                    {
                        if(searchPos == input.size() - 1)
                        {
                            return true;
                        }
                        return false;
                    }
                //}
            }
        }

        return token == 0;
    }

    void CopyNode(const Node & from, Node & to)
    {
        const Node::eType type = from.Type();

        switch(type)
        {
        case Node::SequenceType:
            for(auto it = from.Begin(); it != from.End(); it++)
            {
                const Node & currentNode = (*it).second;
                Node & newNode = to.PushBack();
                CopyNode(currentNode, newNode);
            }
            break;
        case Node::MapType:
            for(auto it = from.Begin(); it != from.End(); it++)
            {
                const Node & currentNode = (*it).second;
                Node & newNode = to[(*it).first];
                CopyNode(currentNode, newNode);
            }
            break;
        case Node::ScalarType:
            to = from.As<std::string>();
            break;
        case Node::None:
            break;
        }
    }

    bool ShouldBeCited(const std::string & key)
    {
        return key.find_first_of("\":{}[],&*#?|-<>=!%@") != std::string::npos;
    }

    void AddEscapeTokens(std::string & input, const std::string & tokens)
    {
        for(auto it = tokens.begin(); it != tokens.end(); it++)
        {
            const char token = *it;
            const std::string replace = std::string("\\") + std::string(1, token);
            size_t found = input.find_first_of(token);
            while(found != std::string::npos)
            {
                input.replace(found, 1, replace);
                found = input.find_first_of(token, found + 2);
            }
        }
    }

    void RemoveAllEscapeTokens(std::string & input)
    {
        size_t found = input.find_first_of("\\");
        while(found != std::string::npos)
        {
            if(found + 1 == input.size())
            {
                return;
            }

            std::string replace(1, input[found + 1]);
            input.replace(found, 2, replace);
            found = input.find_first_of("\\", found + 1);
        }
    }


}
